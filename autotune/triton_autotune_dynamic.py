"""Run one Triton model on several shapes and dtypes with per-input pruning."""

import argparse

import torch
import triton
import triton.language as tl


DEVICE = torch.device("cuda")


def make_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=4 if block < 512 else 8)
        for block in (128, 256, 512)
    ]


def prune_configs(configs, named_args, **kwargs):
    n_elements = named_args["rows"] * named_args["cols"]
    max_block = 256 if named_args["x_ptr"].dtype == torch.float32 else 512
    selected = [
        config for config in configs
        if config.kwargs["BLOCK_SIZE"] <= max_block
        and triton.cdiv(n_elements, config.kwargs["BLOCK_SIZE"]) >= 16
    ]
    return selected or configs[:1]


@triton.autotune(
    configs=make_configs(),
    key=["rows", "cols"],
    prune_configs_by={"early_config_prune": prune_configs},
)
@triton.jit
def relu_kernel(x_ptr, y_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rows * cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        rows, cols = x.shape
        grid = lambda meta: (triton.cdiv(rows * cols, meta["BLOCK_SIZE"]),)
        relu_kernel[grid](x, y, rows, cols)
        return y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compile"), default="compile")
    args = parser.parse_args()

    model = SimpleModel()
    run_model = torch.compile(model, fullgraph=True, dynamic=True) if args.mode == "compile" else model
    compilations = []
    torch._dynamo.on_compile_start(lambda _: compilations.append(1))
    for i, (shape, dtype) in enumerate((
        ((32, 256), torch.float32),
        ((64, 128), torch.float16),
        ((16, 256), torch.float16),
        ((32, 256), torch.float32),
    )):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        y = torch.empty_like(x)
        with torch.inference_mode():
            output = run_model(x, y)
            print(f"run {i}: compile starts = {len(compilations)}")
            torch.testing.assert_close(output, torch.relu(x))
        # selected = prune_configs(model.kernel.configs, {"x_ptr": x, "rows": shape[0], "cols": shape[1]})
        # blocks = [config.kwargs["BLOCK_SIZE"] for config in selected]
        print(f"Run #{i}: {args.mode}, shape={shape}, dtype={dtype}, matches PyTorch")


if __name__ == "__main__":
    main()
