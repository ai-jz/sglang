"""GPU runtime validation for the miles gfx942 wheels (TE, flash-attn, apex).

Runs on an MI300X inside a fresh rocm/sgl-dev:v0.5.20-rocm10-mi30x base with the
freshly built wheels installed, comparing each native op against a torch reference.
"""

import importlib
import sys

import torch

torch.manual_seed(0)
DEV = "cuda"
failures = []


def check(name, out, ref, tol):
    err = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)).item()
    ok = err < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: rel_err={err:.3e} (tol {tol:.0e})", flush=True)
    if not ok:
        failures.append(name)


def section(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
    except Exception as e:  # report every section, then fail at the end
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        failures.append(name)


def env():
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} hip {torch.version.hip} device {props.name} arch {props.gcnArchName}")
    assert props.gcnArchName.startswith("gfx942"), props.gcnArchName


def apex_ops():
    try:
        wgrad = importlib.import_module("fused_weight_gradient_mlp_cuda")
    except ImportError:
        wgrad = importlib.import_module("apex.fused_weight_gradient_mlp_cuda")
    print(f"fused_weight_gradient_mlp_cuda: {wgrad}")
    x = torch.randn(512, 1024, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(512, 768, device=DEV, dtype=torch.bfloat16)
    main_grad = torch.randn(768, 1024, device=DEV, dtype=torch.float32)
    ref = main_grad + g.float().t() @ x.float()
    wgrad.wgrad_gemm_accum_fp32(x, g, main_grad)
    check("apex wgrad_gemm_accum_fp32", main_grad, ref, 1e-2)

    # Warp-reduction kernels: wrong ROCM_WAVEFRONT_SIZE shows up here.
    from apex.normalization import FusedLayerNorm, FusedRMSNorm

    h = torch.randn(64, 4096, device=DEV, dtype=torch.float32, requires_grad=True)
    ln = FusedLayerNorm(4096).to(DEV)
    out = ln(h)
    ref = torch.nn.functional.layer_norm(h, (4096,), ln.weight, ln.bias)
    check("apex FusedLayerNorm fwd", out, ref, 1e-4)
    gy = torch.randn_like(out)
    (dx,) = torch.autograd.grad(out, h, gy)
    (dx_ref,) = torch.autograd.grad(ref, h, gy)
    check("apex FusedLayerNorm bwd", dx, dx_ref, 1e-3)

    rms = FusedRMSNorm(4096).to(DEV)
    out = rms(h)
    ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + rms.eps) * rms.weight
    check("apex FusedRMSNorm fwd", out, ref, 1e-4)


def te_ops():
    import transformer_engine
    import transformer_engine.pytorch as te

    print(f"transformer_engine {transformer_engine.__version__}")
    lin = te.Linear(1024, 768, params_dtype=torch.bfloat16).to(DEV)
    x = torch.randn(256, 1024, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    ref = x.float() @ lin.weight.float().t() + lin.bias.float()
    check("TE Linear fwd", y, ref, 1e-2)
    gy = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, gy)
    check("TE Linear bwd dgrad", dx, gy.float() @ lin.weight.float(), 1e-2)

    norm = te.RMSNorm(4096, params_dtype=torch.float32).to(DEV)
    h = torch.randn(64, 4096, device=DEV, dtype=torch.float32)
    ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    check("TE RMSNorm fwd", norm(h), ref, 1e-4)


def flash_attn_ops():
    import flash_attn
    from flash_attn import flash_attn_func

    print(f"flash_attn {flash_attn.__version__}")
    q, k, v = (torch.randn(2, 256, 8, 128, device=DEV, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    out = flash_attn_func(q, k, v, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), is_causal=True
    ).transpose(1, 2)
    check("flash_attn fwd (causal)", out, ref, 1e-2)
    gy = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), gy)
    dq_r, dk_r, dv_r = torch.autograd.grad(ref, (q, k, v), gy.float())
    check("flash_attn bwd dq", dq, dq_r, 2e-2)
    check("flash_attn bwd dk", dk, dk_r, 2e-2)
    check("flash_attn bwd dv", dv, dv_r, 2e-2)


section("environment", env)
section("apex", apex_ops)
section("transformer_engine", te_ops)
section("flash_attn", flash_attn_ops)
torch.cuda.synchronize()

print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
