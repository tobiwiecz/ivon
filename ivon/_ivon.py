from math import pow
from typing import Callable, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.optim
import torch.distributed as dist
from torch import Tensor


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(torch.optim.Optimizer):
    hessian_approx_methods = (
        'price',
        'gradsq',
    )

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = 'price',
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
        eps: float = 1e-8
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError(
                "Invalid number of MC samples: {}".format(mc_samples)
            )
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not 0.0 < hess_init:
            raise ValueError(
                "Invalid Hessian initialization: {}".format(hess_init)
            )
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(hess_approx))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self.eps = eps
        self._numel, self._device, self._dtype = self._get_param_configs()
        print(f"device is {self._device}")
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        # Set up distributed random generator
        if dist.is_initialized():
            generator_seed = 42 + dist.get_rank()
        else:
            generator_seed = 42
        self._generator = torch.Generator(device=self._device).manual_seed(generator_seed)

        # set initial temporary running averages
        self._reset_samples()
        # init all states
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(
                "Parameters are on different devices: "
                f"{[str(d) for d in devices]}"
            )
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(
                "Parameters are on different dtypes: "
                f"{[str(d) for d in dtypes]}"
            )
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state['count'] = 0
        self.state['avg_grad'] = None
        self.state['avg_nxg'] = None
        self.state['avg_gsq'] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]

            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )
            group["hess"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            ).add(torch.as_tensor(hess_init))

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(
        self, train: bool, param_avg: Tensor, noise: Tensor
    ):
        # TIP 1: More memory-efficient gradient collection
        offset = 0
        grad_tensors = []  # Collect gradient references instead of copying
        
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())
                p.data = param_avg[p_slice].view(p.shape)
                
                # Collect gradient tensors without copying to buffer
                if train:
                    if p.requires_grad and p.grad is not None:
                        grad_tensors.append(p.grad.flatten())
                    else:
                        grad_tensors.append(torch.zeros_like(p.data.flatten()))
                        
                offset += p.numel()
        assert offset == self._numel  # sanity check

        if train and grad_tensors:  # collect grad sample for training
            # Use cat instead of copying to pre-allocated buffer
            current_grad = torch.cat(grad_tensors, dim=0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(
                self.state["avg_grad"], current_grad, count
            )
            
            if self.hess_approx == 'price':
                # Compute noise * grad directly without intermediate buffer
                noise_grad = noise * current_grad
                self.state['avg_nxg'] = _welford_mean(
                    self.state['avg_nxg'], noise_grad, count)
            elif self.hess_approx == 'gradsq':
                # Compute grad^2 directly without intermediate buffer
                grad_squared = current_grad * current_grad
                self.state['avg_gsq'] = _welford_mean(
                    self.state['avg_gsq'], grad_squared, count)

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():  # explicit sync
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        
        # Sync gradient averages
        if self.state["avg_grad"] is not None:
            dist.all_reduce(self.state["avg_grad"])
            self.state["avg_grad"].div_(world_size)
        
        # Sync hessian approximation terms based on method
        if self.hess_approx == 'price' and self.state["avg_nxg"] is not None:
            dist.all_reduce(self.state["avg_nxg"])
            self.state["avg_nxg"].div_(world_size)
        elif self.hess_approx == 'gradsq' and self.state["avg_gsq"] is not None:
            dist.all_reduce(self.state["avg_gsq"])
            self.state["avg_gsq"].div_(world_size)
        
        # Sync count
        if self.state["count"] > 0:
            count_tensor = torch.tensor([self.state["count"]], device=self._device)
            dist.all_reduce(count_tensor)
            self.state["count"] = count_tensor.item()

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        # TIP 2: More memory-efficient parameter sampling
        # Pre-allocate output tensors once
        param_avg_buffer = torch.empty(self._numel, device=self._device, dtype=self._dtype)
        noise_buffer = torch.empty(self._numel, device=self._device, dtype=self._dtype)
        
        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            
            # Generate noise directly into buffer slice to avoid temporary tensors
            group_noise = noise_buffer[offset:offset + gnumel]
            torch.randn(gnumel, device=self._device, dtype=self._dtype, 
                       generator=self._generator, out=group_noise)
            # In-place division to avoid creating temporary tensor
            group_noise.div_((group["ess"] * (group["hess"] + group["weight_decay"])).sqrt())
            
            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                numel = p.numel()
                p_slice = slice(offset + goffset, offset + goffset + numel)
                
                # Store original parameter values in buffer
                param_avg_buffer[p_slice] = p.data.flatten()
                
                # Add noise to parameters in-place
                p.data.add_(group_noise[goffset:goffset + numel].view(p.shape))
                goffset += numel
                
            assert goffset == gnumel  # sanity check
            offset += gnumel
        assert offset == self._numel  # sanity check

        return param_avg_buffer, noise_buffer

    def _update(self):
        # TIP 4: Optimize update method to reduce temporary tensor creation
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            
            if group.get("no_update", False):
                offset += group["numel"]
                continue
            
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            # TIP 4: Work with slices instead of concatenating all parameters
            if self.state["avg_grad"] is not None:
                group["momentum"] = self._new_momentum(
                    self.state["avg_grad"][pg_slice], group["momentum"], b1
                )

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state['avg_gsq'],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            # TIP 4: Update parameters directly without creating intermediate concatenated tensor
            self._update_params_inplace(group, pg_slice, offset)
            offset += group["numel"]
        assert offset == self._numel  # sanity check

    def _update_params_inplace(self, group, pg_slice, global_offset):
        """TIP 4: Update parameters in-place to avoid temporary tensors"""
        lr = group["lr"]
        lr_scale = (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else 1.0
        debias_factor = 1.0 - pow(group["beta1"], float(self.current_step)) if self.debias else 1.0
        
        local_offset = 0
        for p in group["params"]:
            if p is None:
                continue
                
            p_numel = p.numel()
            local_slice = slice(local_offset, local_offset + p_numel)
            
            # Get slices for this parameter
            momentum_slice = group["momentum"][local_slice]
            hess_slice = group["hess"][local_slice]
            
            # Compute update directly without intermediate tensors
            param_flat = p.data.flatten()
            update = torch.clamp(
                (momentum_slice / debias_factor + group["weight_decay"] * param_flat) / 
                (hess_slice + group["weight_decay"] + 1e-12),
                min=-group["clip_radius"], max=group["clip_radius"]
            )
            
            # Apply update in-place
            p.data.sub_(update.view(p.shape), alpha=lr * lr_scale)
            local_offset += p_numel

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == 'price':
            return avg_nxg[pg_slice] * hess if avg_nxg is not None else hess
        elif method == 'gradsq':
            return avg_gsq[pg_slice] if avg_gsq is not None else hess
        else:
            raise NotImplementedError(f'unknown hessian approx.: {method}')

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(
        method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd
    ) -> Tensor:
        f = IVON._get_nll_hess(
            method, hess + wd, avg_nxg, avg_gsq, pg_slice
        ) * ess
        return beta2 * hess + (1.0 - beta2) * f + \
            (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd + 1e-12)

    @staticmethod
    def _new_param_averages(
        param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init
    ) -> Tensor:
        # This method is now unused due to TIP 4 optimization
        return param_avg - lr * torch.clamp(
            (momentum / debias + wd * param_avg) / (hess + wd + 1e-12),
            min=-clip_radius,
            max=clip_radius,
        )