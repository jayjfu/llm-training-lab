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


def split_params(model):
    matrix, adam = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Matrices go to Muon, embedding and lm_head stay on AdamW
        if param.ndim == 2 and not name.endswith("embed_tokens.weight") and not name.endswith("lm_head.weight"):
            matrix.append(param)
        else:
            adam.append(param)
    return matrix, adam


def build_optimizer(model, matrix_lr=0.02, embed_lr=3e-3, weight_decay=0.0):
    matrix, adam = split_params(model)
    muon = Muon(matrix, lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(adam, lr=embed_lr, betas=(0.9, 0.95), weight_decay=0.0)
    return muon, adamw