import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PART 1: Conv (Conv2d + BatchNorm2d + SiLU)
# ============================================================

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================
# PART 2: Bottleneck + C2f
# ============================================================

class Bottleneck(nn.Module):
    """Stack of 2 Convs with optional shortcut connection."""
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels,  out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.add   = shortcut and (in_channels == out_channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """Cross-Stage Partial Bottleneck with 2 convolutions."""
    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=False, expansion=0.5):
        super().__init__()
        self.c   = int(out_channels * expansion)
        # cv1 splits the input into 2 halves along channel dim
        self.cv1 = Conv(in_channels, 2 * self.c, kernel_size=1, stride=1, padding=0)
        self.m   = nn.ModuleList(
            [Bottleneck(self.c, self.c, shortcut) for _ in range(num_bottlenecks)]
        )
        # cv2 merges all collected feature chunks
        self.cv2 = Conv((2 + num_bottlenecks) * self.c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        # FIX: use integer split (not tuple) to avoid shape mismatch
        y = list(x.split(self.c, 1))
        # collect output of each bottleneck
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================
# PART 3: SPPF (Spatial Pyramid Pooling Fast)
# ============================================================

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv8 standard, k=5)."""
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1     = Conv(in_channels, hidden, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # FIX: concat is [x, y1, y2, y3] = 4 * hidden channels, not 4 * out_channels
        self.cv2     = Conv(4 * hidden, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x  = self.cv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# ============================================================
# PART 4: Backbone
# ============================================================

def yolo_parameter(version):
    """Return (depth, width, ratio) scaling factors per model variant."""
    params = {
        'n': (0.33, 0.25, 2.0),
        's': (0.33, 0.50, 2.0),
        'm': (0.67, 0.75, 1.5),
        'l': (1.00, 1.00, 1.0),
        'x': (1.00, 1.25, 1.0),
    }
    if version in params:
        return params[version]
    raise ValueError(f"Unknown version: '{version}'. Choose from {list(params.keys())}")


class Backbone(nn.Module):
    """
    YOLOv8 Backbone — outputs three feature maps at different scales:
        P3  (stride  8) : small   objects  → 80x80  for 640 input
        P4  (stride 16) : medium  objects  → 40x40
        P5  (stride 32) : large   objects  → 20x20
    """
    def __init__(self, version, in_channels=3):
        super().__init__()
        d, w, r = yolo_parameter(version)

        # --- Stem & downsampling convolutions ---
        self.conv_0 = Conv(in_channels,      int(64*w),      kernel_size=3, stride=2, padding=1)  # /2
        self.conv_1 = Conv(int(64*w),        int(128*w),     kernel_size=3, stride=2, padding=1)  # /4  → P1
        self.conv_3 = Conv(int(128*w),       int(256*w),     kernel_size=3, stride=2, padding=1)  # /8  → P2
        self.conv_5 = Conv(int(256*w),       int(512*w),     kernel_size=3, stride=2, padding=1)  # /16 → P3
        self.conv_7 = Conv(int(512*w),       int(512*w*r),   kernel_size=3, stride=2, padding=1)  # /32 → P4

        # --- C2f blocks ---
        self.c2f_2  = C2f(int(128*w),       int(128*w),     num_bottlenecks=int(3*d), shortcut=True)
        self.c2f_4  = C2f(int(256*w),       int(256*w),     num_bottlenecks=int(6*d), shortcut=True)
        self.c2f_6  = C2f(int(512*w),       int(512*w),     num_bottlenecks=int(6*d), shortcut=True)
        self.c2f_8  = C2f(int(512*w*r),     int(512*w*r),   num_bottlenecks=int(3*d), shortcut=True)

        # --- SPPF at deepest scale ---
        self.sppf   = SPPF(int(512*w*r),    int(512*w*r))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        c2f_out_4 = self.c2f_4(x)          # P3 — kept for Neck concat

        x = self.conv_5(c2f_out_4)
        c2f_out_6 = self.c2f_6(x)          # P4 — kept for Neck concat

        x = self.conv_7(c2f_out_6)
        x = self.c2f_8(x)
        sppf_out_9 = self.sppf(x)          # P5

        return c2f_out_4, c2f_out_6, sppf_out_9


# ============================================================
# PART 5: Neck (FPN-style top-down + bottom-up path)
# ============================================================

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.up(x)


class Neck(nn.Module):
    """
    PANet-style neck:
      Top-down path  : SPPF → upsample → concat P4 → C2f
                               upsample → concat P3 → C2f
      Bottom-up path : conv  → concat fused-P4 → C2f
                       conv  → concat fused-P5 → C2f
    Outputs three fused maps fed to the Detect head.
    """
    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_parameter(version)

        # --- Top-down ---
        self.up      = Upsample(scale_factor=2, mode='nearest')

        # After concat(SPPF, P4): channels = 512*w*r + 512*w
        self.c2f_12  = C2f(int(512*w*r) + int(512*w),   int(512*w),   num_bottlenecks=int(3*d), shortcut=False)
        # After concat(c2f_12, P3): channels = 512*w + 256*w
        self.c2f_15  = C2f(int(512*w)   + int(256*w),   int(256*w),   num_bottlenecks=int(3*d), shortcut=False)

        # --- Bottom-up ---
        self.conv_16 = Conv(int(256*w),  int(256*w), kernel_size=3, stride=2, padding=1)
        # After concat(conv_16, c2f_12): channels = 256*w + 512*w
        self.c2f_18  = C2f(int(256*w)   + int(512*w),   int(512*w),   num_bottlenecks=int(3*d), shortcut=False)

        self.conv_19 = Conv(int(512*w),  int(512*w), kernel_size=3, stride=2, padding=1)
        # After concat(conv_19, SPPF): channels = 512*w + 512*w*r
        self.c2f_21  = C2f(int(512*w)   + int(512*w*r), int(512*w*r), num_bottlenecks=int(3*d), shortcut=False)

    def forward(self, c2f_out_4, c2f_out_6, sppf_out_9):
        # --- Top-down path ---
        x            = self.up(sppf_out_9)
        x            = torch.cat((x, c2f_out_6), 1)
        c2f_out_12   = self.c2f_12(x)                   # fused P4

        x            = self.up(c2f_out_12)
        x            = torch.cat((x, c2f_out_4), 1)
        c2f_out_15   = self.c2f_15(x)                   # fused P3 → small obj head

        # --- Bottom-up path ---
        x            = self.conv_16(c2f_out_15)
        x            = torch.cat((x, c2f_out_12), 1)
        c2f_out_18   = self.c2f_18(x)                   # fused P4 → medium obj head

        x            = self.conv_19(c2f_out_18)
        x            = torch.cat((x, sppf_out_9), 1)
        c2f_out_21   = self.c2f_21(x)                   # fused P5 → large obj head

        return c2f_out_15, c2f_out_18, c2f_out_21


# ============================================================
# PART 6: DFL (Distribution Focal Loss helper)
# ============================================================

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss.
    Converts per-bin logits → continuous coordinate via soft-argmax.
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        self.conv.weight.data = torch.arange(c1).view(1, c1, 1, 1).float()
        self.conv.weight.requires_grad_(False)
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape                               # batch, 4*reg_max, anchors
        x = x.view(b, 4, self.c1, a).transpose(2, 1)   # (b, c1, 4, a)
        x = F.softmax(x, dim=1)
        x = self.conv(x).view(b, 4, a)
        return x


# ============================================================
# PART 7: Detect Head (fully decoupled, anchor-free, multi-scale)
# ============================================================

class Detect(nn.Module):
    """
    Decoupled detection head operating on 3 scales independently.

    For each scale:
      - bbox branch : Conv → Conv → Conv2d(4 * reg_max)
      - cls  branch : Conv → Conv → Conv2d(nc)

    This is anchor-free: the model predicts the object center
    directly rather than offsets from anchor boxes.
    """
    def __init__(self, nc=8, ch=()):
        super().__init__()
        self.nc      = nc                   # number of classes
        self.nl      = len(ch)              # number of scales (3)
        self.reg_max = 16                   # DFL bins
        self.no      = nc + self.reg_max * 4

        self.stride  = torch.zeros(self.nl)

        # --- Fully decoupled branches per scale ---
        self.bbox_branch = nn.ModuleList(
            nn.Sequential(
                Conv(x, x, kernel_size=3, stride=1, padding=1),
                Conv(x, x, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(x, 4 * self.reg_max, kernel_size=1, stride=1, padding=0),
            ) for x in ch
        )
        self.cls_branch = nn.ModuleList(
            nn.Sequential(
                Conv(x, x, kernel_size=3, stride=1, padding=1),
                Conv(x, x, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(x, self.nc, kernel_size=1, stride=1, padding=0),
            ) for x in ch
        )
        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        for i in range(self.nl):
            bbox = self.bbox_branch[i](x[i])    # (b, 4*reg_max, h, w)
            cls  = self.cls_branch[i](x[i])     # (b, nc, h, w)
            x[i] = torch.cat((bbox, cls), 1)    # (b, 4*reg_max+nc, h, w)
        return x


# ============================================================
# PART 8: Full YOLOv8 KITTI Model
# ============================================================

class YOLOv8_KITTI(nn.Module):
    """
    Complete YOLOv8 model scaled by 'version' ('n','s','m','l','x').
    Defaults to nc=8 for KITTI classes:
      0:Car  1:Van  2:Truck  3:Pedestrian
      4:Person(sitting)  5:Cyclist  6:Tram  7:Misc
    """
    def __init__(self, version='s', nc=8, in_channels=3):
        super().__init__()
        d, w, r = yolo_parameter(version)

        # 1. Backbone: feature extraction
        self.backbone = Backbone(version, in_channels)

        # 2. Neck: multi-scale feature fusion
        self.neck     = Neck(version)

        # 3. Detect head: one branch-set per scale
        ch = [int(256*w), int(512*w), int(512*w*r)]
        self.detect   = Detect(nc=nc, ch=ch)

    def forward(self, x):
        # Backbone → 3 feature maps
        c2f_out_4, c2f_out_6, sppf_out_9 = self.backbone(x)

        # Neck → 3 fused feature maps
        p3, p4, p5 = self.neck(c2f_out_4, c2f_out_6, sppf_out_9)

        # Detect head → predictions at 3 scales
        return self.detect([p3, p4, p5])


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == '__main__':
    dummy = torch.zeros(1, 3, 640, 640)

    print(f"{'Version':<10} {'Params (M)':<14} {'P3 shape':<24} {'P4 shape':<24} {'P5 shape'}")
    print("-" * 95)

    for v in ['n', 's', 'm', 'l', 'x']:
        model = YOLOv8_KITTI(version=v, nc=8)
        model.eval()
        with torch.no_grad():
            outs = model(dummy)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        shapes = [str(tuple(o.shape)) for o in outs]
        print(f"  '{v}'      {params:<14.2f} {shapes[0]:<24} {shapes[1]:<24} {shapes[2]}")

    print("\nSanity Check✅ Output channels: 72 = 4x16 (DFL bbox bins) + 8 (KITTI classes)")