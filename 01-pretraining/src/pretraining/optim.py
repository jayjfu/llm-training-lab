import torch


def newton_schulz(grad, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    ortho = grad.bfloat16()
    ortho = ortho / (ortho.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        ortho = ortho.mT
    for _ in range(steps):
        gram = ortho @ ortho.mT
        ortho = a * ortho + (b * gram + c * gram @ gram) @ ortho
    if transposed:
        ortho = ortho.mT
    return ortho


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            momentum = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.grad)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(param.grad)
                update = param.grad.add(buffer, alpha=momentum) if group["nesterov"] else buffer
                ortho = newton_schulz(update, steps=group["ns_steps"]).to(param.dtype)
                scaled_lr = group["lr"] * max(1.0, param.size(0) / param.size(1)) ** 0.5
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(ortho, alpha=-scaled_lr)


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add_(param.grad, alpha=1 - beta1).sign_()
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update, alpha=-group["lr"])
                exp_avg.mul_(beta2).add_(param.grad, alpha=1 - beta2)


def split_params(model):
    matrix, other = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.startswith(("wte", "wpe", "lm_head")):
            matrix.append(param)
        else:
            other.append(param)  # embeddings, head, norms, biases
    return matrix, other


def build_optimizers(model, optimizer, lr, muon_lr, weight_decay):
    matrix, other = split_params(model)
    if optimizer == "muon":
        optimizers = [
            Muon(matrix, lr=muon_lr),
            torch.optim.AdamW(other, lr=lr, betas=(0.9, 0.95), weight_decay=0.0),
        ]
    else:
        groups = [{"params": matrix, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}]
        if optimizer == "lion":
            optimizers = [Lion(groups, lr=lr)]
        else:
            optimizers = [torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)]
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers