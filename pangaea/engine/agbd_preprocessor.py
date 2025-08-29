"""
Custom preprocessing utilities for the AGBD dataset.

This module implements a variant of the standard ``ResizeToEncoder``
transform that keeps track of the centre pixel when upsampling a
25×25 patch to the input size expected by a given encoder.  The
default ``ResizeToEncoder`` in PANGAEA resizes the image (and
optionally the target) but does not update any metadata.  For the
AGBD dataset we supervise only the centre pixel; therefore when
resizing a patch from its native resolution (25×25) to a larger
encoder input size (e.g. 256×256) we must update the ``center_pixel_yx``
entry in the ``metadata`` dictionary so that downstream losses and
metrics operate on the correct pixel.  This preprocessor performs
that update.

The transformation supports resizing both the image and the target.  It
inherits from ``pangaea.engine.data_preprocessor.ResizeToEncoder`` and
adds a metadata update step.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, Any

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF

from pangaea.engine.data_preprocessor import ResizeToEncoder


class ResizeToEncoderWithCenter(ResizeToEncoder):
    """Resize the input image and target to the encoder input size and update centre coordinates.

    PANGAEA encoders often require a fixed input size (e.g. 256).  When
    training with the AGBD dataset we start with 25×25 context
    patches, so a resize is necessary for most encoders.  This
    class extends ``ResizeToEncoder`` by updating the
    ``center_pixel_yx`` entry in the metadata.  The new centre
    coordinate is computed by scaling the old coordinate with the
    ratio between the new size and the original size, following
    ``new_center = round((old_center + 0.5) * scale - 0.5)``.  This
    formula ensures that the centre pixel aligns with the geometrical
    centre of the resized patch.
    """

    def __init__(
        self,
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        resize_target: bool = True,
        **meta,
    ) -> None:
        """Initialise the resizer.

        Args:
            interpolation: interpolation mode used for resizing the image.
            antialias: whether to apply anti‑aliasing.  Defaults to ``True``.
            resize_target: whether to resize the target map.  For AGBD we
                always resize the target so that its spatial dimensions
                match the resized image.  Defaults to ``True``.
            **meta: metadata from the dataset configuration.  Must
                include ``encoder_input_size`` and ``data_img_size``.
        """
        # Force resizing of target by default, because the
        # downstream code expects target and image to have the same
        # spatial dimensions.
        super().__init__(
            interpolation=interpolation,
            antialias=antialias,
            resize_target=resize_target,
            **meta,
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resize images and targets and update the centre coordinates.

        This method first delegates resizing to the parent class and then
        updates the ``metadata['center_pixel_yx']`` entry to reflect
        the new location of the centre pixel in the resized patch.  The
        update uses the ratio between the encoder input size and the
        original data size recorded in the metadata.  The centre
        coordinate is rounded to the nearest integer to avoid bias.

        Args:
            data: a dictionary with keys ``'image'``, ``'target'`` and
                ``'metadata'``.  The ``'image'`` value is a dictionary
                mapping modality names to tensors of shape ``(C,T,H,W)``;
                the ``'target'`` value is a tensor of shape ``(H,W)``; and
                ``'metadata'`` is a list of metadata dictionaries (one per
                sample) or a single dictionary for unbatched data.

        Returns:
            The resized data with updated metadata.
        """
        # Delegate resizing of image and target to parent
        data = super().__call__(data)

        # Retrieve sizes from metadata: original size and new size
        # ``data_img_size`` has been updated by the parent class to the
        # new size; ``meta['dataset_img_size']`` holds the original size.
        # However, when this transform is part of a pipeline, the meta
        # dictionary passed at construction time contains both values.
        # We obtain them from self.size (target size) and the stored
        # original size in meta.  At call time ``metadata`` stores
        # per‑sample centre coordinates.
        new_size = self.size[0] if isinstance(self.size, Sequence) else self.size
        # Update the centre coordinate for each sample in metadata
        metadata_list = data.get('metadata', None)
        if metadata_list is None:
            return data
        # Determine the old image size.  It may be stored in
        # ``self.data_img_size`` attribute inherited from Resize (set via
        # meta), but to be safe read it from the first metadata entry if
        # present.
        # For unbatched data metadata_list is a dict, wrap it in a list
        if isinstance(metadata_list, dict):
            meta_iter = [metadata_list]
        else:
            meta_iter = metadata_list
        # Determine original patch dimension from first sample
        if not hasattr(self, 'orig_img_size'):
            # Attempt to infer from meta or fall back to length of
            # target height before resize
            # The dataset provides centre coordinates relative to the
            # original 25×25 patch, so the original size can be
            # deduced from twice the centre index plus one.
            first_meta = meta_iter[0]
            cy, cx = first_meta.get('center_pixel_yx', (new_size // 2, new_size // 2))
            # The original size is assumed square
            self.orig_img_size = int(2 * cy + 1)
        scale = new_size / float(self.orig_img_size)
        for meta in meta_iter:
            if 'center_pixel_yx' in meta:
                y, x = meta['center_pixel_yx']
                # Compute new centre using continuous coordinate system
                new_y = int(round((y + 0.5) * scale - 0.5))
                new_x = int(round((x + 0.5) * scale - 0.5))
                meta['center_pixel_yx'] = (new_y, new_x)
        return data

    def update_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Update the meta dictionary for subsequent preprocessors.

        The parent class already updates ``data_img_size`` to the new
        size; we additionally record the new size in
        ``data_img_size`` and leave the original
        ``dataset_img_size`` unchanged.  We do not modify band
        statistics here.

        Args:
            meta: dictionary storing dataset and encoder metadata.

        Returns:
            The updated meta dictionary.
        """
        # Call parent update_meta to update data_img_size
        meta = super().update_meta(meta)
        # Note: centre coordinates are stored in metadata at sample level;
        # they are updated during __call__, not here.
        return meta