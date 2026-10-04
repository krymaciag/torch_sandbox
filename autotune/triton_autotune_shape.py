"""Autotune a Triton kernel by tensor shape in eager or compile mode."""

import argparse

import torch
import triton
import triton.language as tl


DEVICE = torch.device("cuda")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["rows", "cols"],
)
@triton.jit
def relu_kernel(x_ptr, y_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def triton_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    grid = lambda meta: (triton.cdiv(rows * cols, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, y, rows, cols)
    return y


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.5))

    def forward(self, x, y):
        return triton_relu(x * self.scale, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compile"), default="compile")
    args = parser.parse_args()

    model = SimpleModel().to(DEVICE).eval()
    run_model = torch.compile(model, fullgraph=True) if args.mode == "compile" else model
    with torch.inference_mode():
        for shape in ((32, 256), (64, 128), (32, 256)):
            x = torch.randn(shape, device=DEVICE)
            y = torch.empty_like(x)
            output = run_model(x, y)
            torch.testing.assert_close(output, torch.relu(x * model.scale))
            print(f"{args.mode}: shape={shape}, numel={x.numel()}, matches PyTorch")


if __name__ == "__main__":
    main()
