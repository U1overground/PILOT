# models/pidnet_cl.py
# Minimal continual-learning add-on for PIDNet: adds a small parallel D-branch
# and a binary head for a single new class, while keeping the original 15-class
# path frozen and unchanged.

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the base PIDNet and utilities
from .pidnet import PIDNet, bn_mom, algc
from .model_utils import BasicBlock, Bottleneck, segmenthead


class PIDNetCL(PIDNet):
    """
    PIDNet with a minimal continual-learning add-on.

    - Keeps the original 15-class segmentation path frozen and unchanged.
    - Adds a parallel partial D-branch (layer3_d_new, layer4_d_new, diff3_new, diff4_new)
      that taps the network at the same resolution as the official boundary head.
    - Adds a small binary segmentation head `seghead_new` with 2 outputs: {new_class, background}.

    The forward path mirrors the original PIDNet forward, and additionally produces
    logits for the new class from the new branch. Fusion with P/I is intentionally
    omitted to keep it minimal.
    """

    def __init__(self, m: int = 2, n: int = 3, num_classes: int = 15,
                 planes: int = 32, ppm_planes: int = 96, head_planes: int = 128,
                 augment: bool = True):
        super().__init__(m=m, n=n, num_classes=num_classes,
                         planes=planes, ppm_planes=ppm_planes,
                         head_planes=head_planes, augment=augment)

        # Build the new minimal D-path modules to mirror the temp_d tap point
        if m == 2:
            # Small model settings (matches base PIDNet for 'pidnet_s')
            in_d3 = planes * 2
            mid_d3 = planes
            out_d4_in = planes
            out_planes_d = planes * 2
            self.layer3_d_new = self._make_single_layer(BasicBlock, in_d3, mid_d3)
            self.layer4_d_new = self._make_layer(Bottleneck, out_d4_in, out_d4_in, 1)
            self.diff3_new = nn.Sequential(
                nn.Conv2d(planes * 4, mid_d3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_d3, momentum=bn_mom),
            )
            self.diff4_new = nn.Sequential(
                nn.Conv2d(planes * 8, out_planes_d, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_planes_d, momentum=bn_mom),
            )
        else:
            # Medium/Large variants mirror the alternative D-branch widths
            ch = planes * 2
            self.layer3_d_new = self._make_single_layer(BasicBlock, ch, ch)
            self.layer4_d_new = self._make_single_layer(BasicBlock, ch, ch)
            self.diff3_new = nn.Sequential(
                nn.Conv2d(planes * 4, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch, momentum=bn_mom),
            )
            self.diff4_new = nn.Sequential(
                nn.Conv2d(planes * 8, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch, momentum=bn_mom),
            )

        # Two-channel head for {new_class, background} sitting on the new D branch
        self.seghead_new = segmenthead(planes * 2, head_planes, 2)

        # Init the new modules only
        for mmod in [self.layer3_d_new, self.layer4_d_new, self.diff3_new, self.diff4_new, self.seghead_new]:
            for m in mmod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # By default, freeze the base network so training only affects new modules
        self.freeze_base_parameters()

    # --------------------------- Utils ---------------------------
    def freeze_base_parameters(self):
        for name, p in self.named_parameters():
            # Unfreeze only the newly added modules
            if any(tag in name for tag in [
                'layer3_d_new', 'layer4_d_new', 'diff3_new', 'diff4_new', 'seghead_new'
            ]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def new_parameters(self):
        """Convenience to fetch trainable new-module params only."""
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def load_frozen_from_checkpoint(self, ckpt_path: str, key: str = None):
        """
        Load the pretrained 15-class weights into the base part of this model.
        The new modules remain randomly initialized.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt if isinstance(ckpt, dict) and 'state_dict' not in ckpt else ckpt.get('state_dict', ckpt)
        # Strip common wrappers like 'module.'
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[len('module.') :]
            # Skip newly added keys
            if any(tag in nk for tag in ['layer3_d_new', 'layer4_d_new', 'diff3_new', 'diff4_new', 'seghead_new']):
                continue
            new_state[nk] = v
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        return missing, unexpected

    # --------------------------- Forward variants ---------------------------
    def forward_with_new(self, x: torch.Tensor, upsample_new: bool = True) -> Dict[str, Any]:
        """
        Run the standard PIDNet forward to get the frozen 15-class logits, and in parallel
        build the new D-branch features at the same resolution as the boundary head
        to produce new binary logits for the extra class.

        Returns a dict with:
          - 'logits_15': [B, 15, H, W]    (frozen)
          - 'logits_new_2ch': [B, 2, H, W] (two-channel; new-class vs background)
          - 'lowres_new': [B, 2, H/8, W/8] (pre-upsample logits for debugging)
        """
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        # Shared stem and early layers (mirror base forward)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        # Prepare branches
        x_p = self.layer3_(x)          # P branch early
        x_d_base = self.layer3_d(x)    # Original D branch start
        x_d_new = self.layer3_d_new(x) # New D branch start

        x_m = self.relu(self.layer3(x))
        x_p = self.pag3(x_p, self.compression3(x_m))

        # Add diff3 to both D paths
        add3 = F.interpolate(self.diff3(x_m), size=[height_output, width_output], mode='bilinear', align_corners=algc)
        x_d_base = x_d_base + add3
        add3_new = F.interpolate(self.diff3_new(x_m), size=[height_output, width_output], mode='bilinear', align_corners=algc)
        x_d_new = x_d_new + add3_new

        # Continue
        x_p = self.layer4_(self.relu(x_p))
        x_d_base = self.layer4_d(self.relu(x_d_base))
        x_d_new = self.layer4_d_new(self.relu(x_d_new))

        x_m = self.relu(self.layer4(x_m))
        add4 = F.interpolate(self.diff4(x_m), size=[height_output, width_output], mode='bilinear', align_corners=algc)
        x_d_base = x_d_base + add4
        add4_new = F.interpolate(self.diff4_new(x_m), size=[height_output, width_output], mode='bilinear', align_corners=algc)
        x_d_new = x_d_new + add4_new

        # Base path proceeds to fusion and final 15-class logits
        if self.augment:
            temp_p = x_p
            temp_d = x_d_base  # not used here but kept for parity

        x_p = self.layer5_(self.relu(x_p))
        x_d_base = self.layer5_d(self.relu(x_d_base))
        x_spp = F.interpolate(self.spp(self.layer5(x_m)), size=[height_output, width_output], mode='bilinear', align_corners=algc)
        logits_15_lowres = self.final_layer(self.dfm(x_p, x_spp, x_d_base))  # [B, 15, H/8, W/8]
        logits_15 = F.interpolate(logits_15_lowres, scale_factor=8, mode='bilinear', align_corners=algc)

        # New binary head sits at the pre-layer5_d stage of new D-branch
        logits_new_lowres = self.seghead_new(x_d_new)  # [B, 2, H/8, W/8]
        logits_new = F.interpolate(logits_new_lowres, scale_factor=8, mode='bilinear', align_corners=algc) if upsample_new else logits_new_lowres

        return {
            'logits_15': logits_15,
            'logits_new_2ch': logits_new,
            'lowres_new': logits_new_lowres,
        }

    # Keep the original forward for backward compatibility
    def forward(self, x: torch.Tensor):
        return super().forward(x)
