

PR for prune_configs:

https://github.com/pytorch/pytorch/pull/142207


Issues:
- dynamic shapes perf_model should not depends on symbolic values



Are function from prune_configs_by call per set of keys ? 


In eager mode, yes: early_config_prune runs when Triton sees a new combination of autotune key values and tensor dtypes. A repeat call with the same combination uses the cached choice. perf_model runs for each remaining candidate only if Triton needs to narrow them to top_k. Triton source
Under torch.compile, it is different: PyTorch evaluates the pruning functions while tracing a graph, so they can run again on a graph recompile. Their calls are not guaranteed to be once per Triton key. That is the path shown in your installed [PyTorch wrapper (line 2143)](/home/arki/miniconda3/lib/python3.13/site-packages/torch/_higher_order_ops/triton_kernel_wrap.py:2143).