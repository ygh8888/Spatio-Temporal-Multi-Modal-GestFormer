"""
GestFormer 환경 수정 스크립트
------------------------------
실행 방법 (저장소 루트에서):
    python apply_fixes.py

적용 내용:
  1. models/backbones/resnet.py  — torchvision.models.utils import 수정
  2. models/backbones/vgg.py     — 동일
  3. models/backbones/r3d.py     — 동일
  4. models/attention.py         — DWT __init__ 이동 + .to(x.device) 수정
  5. models/model_utilizer.py    — DataParallel MIG 가드 추가
  6. test.py                     — 선택 패키지 import 안전 처리
  7. hyperparameters/**/*.json   — data_path 실제 경로로 변경
"""

import re
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "src_gestformer")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✅  {os.path.relpath(path)}")


def patch_file(rel_path, replacements):
    """replacements: list of (old_str, new_str) tuples"""
    path = os.path.join(BASE, rel_path)
    if not os.path.exists(path):
        print(f"  ⚠️  파일 없음, 건너뜀: {rel_path}")
        return
    content = read(path)
    for old, new in replacements:
        if old not in content:
            print(f"  ⚠️  대상 문자열 없음 (이미 적용됐거나 버전 차이): {rel_path}")
            print(f"       찾는 내용: {old[:60].strip()!r}")
            continue
        content = content.replace(old, new, 1)
    write(path, content)


# ─────────────────────────────────────────────────────────────
# 1‒3. torchvision.models.utils → torch.hub
# ─────────────────────────────────────────────────────────────
OLD_IMPORT = "from torchvision.models.utils import load_state_dict_from_url"
NEW_IMPORT = "from torch.hub import load_state_dict_from_url"

print("\n[1/7] resnet.py — import 경로 수정")
patch_file("models/backbones/resnet.py", [(OLD_IMPORT, NEW_IMPORT)])

print("[2/7] vgg.py — import 경로 수정")
patch_file("models/backbones/vgg.py", [(OLD_IMPORT, NEW_IMPORT)])

print("[3/7] r3d.py — import 경로 수정")
patch_file("models/backbones/r3d.py", [(OLD_IMPORT, NEW_IMPORT)])

# ─────────────────────────────────────────────────────────────
# 4. attention.py
#    (a) SSL.__init__  — DWT 모듈을 인스턴스 변수로 선언
#    (b) SSL.forward   — self.dwt / self.idwt 사용 + .cuda() 제거
#    (c) EncoderSelfAttention.forward — .cuda(device=0) → .to(x.device)
# ─────────────────────────────────────────────────────────────
print("[4/7] attention.py — DWT 초기화 위치 수정 + 디바이스 지정 수정")

ATT_PATH = "models/attention.py"

# (a) SSL.__init__ 마지막 conv 줄 뒤에 DWT 모듈 추가
OLD_INIT = (
    "        self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)"
    "#conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)\n"
)

# conv_cat 줄은 파일에서 한 줄이므로 정확히 찾는다
OLD_CONV_CAT = (
    "        self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)"
    "#conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)"
)
NEW_CONV_CAT = (
    "        self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)"
    "#conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)\n"
    "        # DWT 모듈: __init__에서 한 번만 생성 (MIG 환경 메모리 절약)\n"
    "        self.dwt = DWTForward(J=1, mode='zero', wave='db3')\n"
    "        self.idwt = DWTInverse(wave='db3', mode='zero')"
)
patch_file(ATT_PATH, [(OLD_CONV_CAT, NEW_CONV_CAT)])

# (b) SSL.forward — DWT 재생성 코드 제거 후 self.* 사용
OLD_FORWARD_DWT = (
    "        aa =  DWTForward(J=1, mode='zero', wave='db3').cuda(device=0)\n"
    "        yl, yh = aa(x)"
)
NEW_FORWARD_DWT = "        yl, yh = self.dwt(x)"

OLD_FORWARD_IDWT = (
    "        ifm = DWTInverse(wave='db3', mode='zero').cuda(device=0)\n"
    "        Y = ifm((conv_rec1, rec_yh))"
)
NEW_FORWARD_IDWT = "        Y = self.idwt((conv_rec1, rec_yh))"

patch_file(ATT_PATH, [
    (OLD_FORWARD_DWT,  NEW_FORWARD_DWT),
    (OLD_FORWARD_IDWT, NEW_FORWARD_IDWT),
])

# (c) EncoderSelfAttention — .cuda(device=0) → .to(x.device)
OLD_SINUSOID = ".expand(x.shape).cuda(device=0)"
NEW_SINUSOID = ".expand(x.shape).to(x.device)"
patch_file(ATT_PATH, [(OLD_SINUSOID, NEW_SINUSOID)])

# ─────────────────────────────────────────────────────────────
# 5. model_utilizer.py — DataParallel MIG 가드
# ─────────────────────────────────────────────────────────────
print("[5/7] model_utilizer.py — DataParallel MIG 가드 추가")

OLD_DP = "        net = nn.DataParallel(net, device_ids=self.configer.get('gpu')).to(self.device)"
NEW_DP = (
    "        # MIG 환경: device_count()를 초과하는 gpu_id 제거\n"
    "        available = torch.cuda.device_count()\n"
    "        gpu_ids = [g for g in self.configer.get('gpu') if g < available]\n"
    "        net = nn.DataParallel(net, device_ids=gpu_ids).to(self.device)"
)
patch_file("models/model_utilizer.py", [(OLD_DP, NEW_DP)])

# ─────────────────────────────────────────────────────────────
# 6. test.py — 선택 패키지 import 안전 처리
# ─────────────────────────────────────────────────────────────
print("[6/7] test.py — 선택 패키지 import 안전 처리")

OLD_IMPORTS = (
    "from torchstat import stat\n"
    "import time\n"
    "import torchsummary\n"
    "from fvcore.nn import FlopCountAnalysis"
)
NEW_IMPORTS = (
    "import time\n"
    "try:\n"
    "    from torchstat import stat\n"
    "    import torchsummary\n"
    "    from fvcore.nn import FlopCountAnalysis\n"
    "except ImportError:\n"
    "    stat = torchsummary = FlopCountAnalysis = None"
)
patch_file("test.py", [(OLD_IMPORTS, NEW_IMPORTS)])

# ─────────────────────────────────────────────────────────────
# 7. JSON 설정 파일 — data_path 실제 경로로 변경
# ─────────────────────────────────────────────────────────────
print("[7/7] hyperparameters/*.json — data_path 수정")

import json

JSON_PATHS = {
    "hyperparameters/Briareo/train.json":     "/data/Briareo",
    "hyperparameters/Briareo/test.json":      "/data/Briareo",
    "hyperparameters/NVGestures/train.json":  "/data/NVGestures",
    "hyperparameters/NVGestures/test.json":   "/data/NVGestures",
}

for rel, new_path in JSON_PATHS.items():
    full = os.path.join(BASE, rel)
    if not os.path.exists(full):
        print(f"  ⚠️  파일 없음: {rel}")
        continue
    with open(full, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["data"]["data_path"] = new_path
    with open(full, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"  ✅  {rel}  →  data_path: {new_path}")

print("\n✨  모든 수정 완료!\n")
print("다음 명령어로 GitHub에 반영하세요:")
print("  git add .")
print("  git commit -m 'fix: MIG/CUDA12 호환성 수정 및 data_path 설정'")
print("  git push origin main")
