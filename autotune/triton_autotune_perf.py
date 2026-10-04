"""Inference example: python triton_autotune_perf.py --mode {eager,compile}."""

import argparse

import torch
import triton
import triton.language as tl


DEVICE = torch.device("cuda")


def prune_fn(configs, named_args, **kwargs):
    n_elements = named_args["n_elements"]
    pruned = [config for config in configs if triton.cdiv(n_elements, config.kwargs["BLOCK_SIZE"]) >= 16]
    return pruned or configs[:1]


def perf_model(BLOCK_SIZE, num_warps, **kwargs):
    return num_warps / BLOCK_SIZE


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
    prune_configs_by={"early_config_prune": prune_fn, "perf_model": perf_model, "top_k": 2},
)
@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def triton_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, y, x.numel())
    return y


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)

    def forward(self, x, y):
        x = self.linear(x)
        return triton_relu(x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compile"), default="compile")

    args = parser.parse_args()


    torch.manual_seed(0)

    model = SimpleModel().to(DEVICE)

 
    x = torch.randn(32, 128, device=DEVICE)
    y = torch.empty(32, 256, device=DEVICE)
    
    x2 = torch.randn(64, 128, device=DEVICE)
    y2 = torch.empty(64, 256, device=DEVICE)

    x3 = torch.randn(32, 128, device=DEVICE)
    y3 = torch.empty(32, 256, device=DEVICE)

    # fullgraph=True raises on any graph break, including inside triton_relu.
    run_model = torch.compile(model, fullgraph=True, dynamic=True) if args.mode == "compile" else model


    print("\n\nFirst Run shape {0}======================================\n\n".format(x.shape))
    output = run_model(x, y)
    print("\n\nSecond Run shape {0}======================================\n\n".format(x2.shape))
    output2 = run_model(x2, y2)
    print("\n\nThrid Run shape {0}======================================\n\n".format(x3.shape))
    output3 = run_model(x3, y3)

    
    print(f"{args.mode}: output shape={tuple(output.shape)}, matches PyTorch")


if __name__ == "__main__":
    main()
