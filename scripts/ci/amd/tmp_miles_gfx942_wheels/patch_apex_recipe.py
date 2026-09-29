"""Relax build_apex_wheel.py's rocminfo check for ROCm 10.

ROCm 10 rocminfo also lists the generic ISA ``amdgcn-amd-amdhsa--gfx9-4-generic``, which
the recipe's ``\\bgfx[0-9a-f]+\\b`` scan reads as a second arch ``gfx9``. Check instead
exactly what Apex's op_builder reads: GPU agent names, and the first ``Wavefront Size``.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
start = src.index("def validate_gpu(arch, rocminfo):")
end = src.index("def check_environment(arch):")
new = '''def validate_gpu(arch, rocminfo):
    detected = set(re.findall(r"^\\s*Name:\\s+(gfx[0-9a-f]+)\\s*$", rocminfo, re.M))
    if detected != {arch}:
        raise RuntimeError(
            f"Requested {arch}, but rocminfo reports {sorted(detected)}. "
            "Use matching hardware and expose only that GPU architecture."
        )
    waves = re.findall(r"Wavefront Size:\\s+(\\d+)", rocminfo)
    # Apex's op_builder uses the first match; every GPU agent must agree.
    if not waves or waves[0] != "64" or set(waves) - {"0"} != {"64"}:
        raise RuntimeError(f"Expected wavefront size 64 for {arch}, got {waves}")
    print(f"rocminfo: GPU agents {sorted(detected)}, wavefront sizes {waves}", flush=True)


'''
path.write_text(src[:start] + new + src[end:])
