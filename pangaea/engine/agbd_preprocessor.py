from __future__ import annotations

from typing import Dict, Any, List, Sequence

from pangaea.engine.data_preprocessor import ResizeToEncoder


class ResizeToEncoderWithCenter(ResizeToEncoder):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["resize_target"] = False
        super().__init__(*args, **kwargs)
        self._logged_once = False

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" not in data or "metadata" not in data:
            raise RuntimeError("data must contain 'image' and 'metadata'")
        data = super().__call__(data)

        metas: List[Dict[str, Any]] = (
            data["metadata"]
            if isinstance(data["metadata"], list)
            else [data["metadata"]]
        )
        if len(metas) == 0:
            raise RuntimeError("metadata list is empty")

        if isinstance(self.size, Sequence):
            new_h = int(self.size[0])
            new_w = int(self.size[1]) if len(self.size) > 1 else int(self.size[0])
        else:
            new_h = new_w = int(self.size)

        for m in metas:
            if "_orig_hw" not in m or "center_pixel_yx" not in m:
                raise RuntimeError(
                    "metadata must contain '_orig_hw' and 'center_pixel_yx'"
                )
            oh, ow = m["_orig_hw"]
            cy, cx = m["center_pixel_yx"]
            sy = new_h / float(oh)
            sx = new_w / float(ow)
            ny = int(round((cy + 0.5) * sy - 0.5))
            nx = int(round((cx + 0.5) * sx - 0.5))
            if not (0 <= ny < new_h and 0 <= nx < new_w):
                raise RuntimeError(
                    f"resized center out of bounds: {(ny, nx)} not in [0,{new_h})x[0,{new_w})"
                )
            m["center_pixel_yx"] = (ny, nx)

        if not self._logged_once:
            k = next(iter(data["image"]))
            x = data["image"][k]
            if x.ndim == 4:
                _, _, H, W = x.shape
            elif x.ndim == 5:
                _, _, _, H, W = x.shape
            else:
                raise RuntimeError(f"unexpected image tensor shape {tuple(x.shape)}")
            cy, cx = metas[0]["center_pixel_yx"]
            oh, ow = metas[0]["_orig_hw"]
            print(
                f"[AGBD-Prep] resized to {(H, W)} from {(oh, ow)}; center -> {(cy, cx)}"
            )
            self._logged_once = True

        return data
