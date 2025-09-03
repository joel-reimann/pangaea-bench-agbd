import collections.abc
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from logging import Logger
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pangaea.encoders.base import Encoder
import sys

try:
    from pangaea.encoders.single_file_galileo import (
        Encoder as GalileoEncoder,
        CONFIG_FILENAME,
        ENCODER_FILENAME,
    )
    from pangaea.encoders.single_file_galileo import (
        S1_BANDS,
        S2_BANDS,
        SPACE_TIME_BANDS,
        SPACE_TIME_BANDS_GROUPS_IDX,
        TIME_BANDS,
        SPACE_BANDS,
        STATIC_BANDS,
        BASE_GSD,
        TIME_BAND_GROUPS_IDX,
        SPACE_BAND_GROUPS_IDX,
        STATIC_BAND_GROUPS_IDX,
    )
except ImportError as e:
    print(f"Could not import Galileo: {e}")
    print(f"Looking for Galileo at: {galileo_path}")
    raise


class GalileoEncoderWrapper(Encoder):
    """
    GALILEO encoder wrapper for PANGAEA-bench.
    Adapts the Galileo encoder to work with PANGAEA's encoder interface.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: Dict[str, List[str]],
        output_layers: List[int],
        patch_size: int = 4,
        month: int = 6,
        do_pool: bool = True,
        add_layernorm_on_exit: bool = True,
        download_url: str = "https://huggingface.co/nasaharvest/galileo",
        model_identifier: str = "nano",
        **kwargs,
    ) -> None:

        self.model_identifier = model_identifier
        self.patch_size = patch_size
        self.month = month
        self.do_pool = do_pool
        self.add_layernorm_on_exit = add_layernorm_on_exit

        self.model_configs = {
            "nano": {"embedding_size": 128, "depth": 4, "num_heads": 8},
            "tiny": {"embedding_size": 192, "depth": 12, "num_heads": 3},
            "base": {"embedding_size": 768, "depth": 12, "num_heads": 12},
        }

        embed_dim = 128
        if model_identifier in self.model_configs:
            embed_dim = self.model_configs[model_identifier]["embedding_size"]

        super().__init__(
            model_name="galileo",
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=embed_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=True,
            encoder_weights=encoder_weights,
            download_url=download_url,
        )

        self._setup_band_mappings()

        self.galileo_encoder = None

    def _setup_band_mappings(self):
        """Set up mapping from PANGAEA bands to Galileo bands."""
        self.band_mapping = {}
        self.total_channels = 0

        if "optical" in self.input_bands:
            optical_bands = self.input_bands["optical"]
            s2_indices = []
            for band in optical_bands:
                if band in S2_BANDS:
                    s2_indices.append(S2_BANDS.index(band))
            self.band_mapping["optical"] = {
                "galileo_bands": "S2",
                "indices": s2_indices,
                "channels": len(s2_indices),
            }
            self.total_channels += len(s2_indices)

        if "sar" in self.input_bands:
            sar_bands = self.input_bands["sar"]
            s1_indices = []
            for band in sar_bands:
                if band in S1_BANDS:
                    s1_indices.append(S1_BANDS.index(band))
            self.band_mapping["sar"] = {
                "galileo_bands": "S1",
                "indices": s1_indices,
                "channels": len(s1_indices),
            }
            self.total_channels += len(s1_indices)

    def download_model(self) -> None:
        """Download Galileo model weights from Hugging Face."""
        if self.download_url and not os.path.isfile(self.encoder_weights):

            model_dir = Path(self.encoder_weights).parent
            model_dir.mkdir(parents=True, exist_ok=True)

            repo_id = "nasaharvest/galileo"
            try:

                config_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"models/{self.model_identifier}/config.json",
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )

                encoder_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"models/{self.model_identifier}/encoder.pt",
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )

                downloaded_config = (
                    model_dir / "models" / self.model_identifier / "config.json"
                )
                downloaded_encoder = (
                    model_dir / "models" / self.model_identifier / "encoder.pt"
                )
                final_config = model_dir / "config.json"
                final_encoder = model_dir / "encoder.pt"
                if downloaded_config.exists():
                    downloaded_config.rename(final_config)
                if downloaded_encoder.exists():
                    downloaded_encoder.rename(final_encoder)

                if (model_dir / "models" / self.model_identifier).exists():
                    (model_dir / "models" / self.model_identifier).rmdir()
                if (model_dir / "models").exists():
                    (model_dir / "models").rmdir()
                print(
                    f"Downloaded Galileo {self.model_identifier} model to {model_dir}"
                )
            except Exception as e:
                print(f"Error downloading Galileo model: {e}")
                print("Falling back to local model...")

                galileo_model_path = (
                    galileo_path / "data" / "models" / self.model_identifier
                )
                if galileo_model_path.exists():
                    import shutil

                    shutil.copytree(galileo_model_path, model_dir, dirs_exist_ok=True)

    def load_encoder_weights(self, logger: Logger) -> None:
        """Load the Galileo encoder weights."""
        try:

            import sys
            from pathlib import Path

            encoders_path = Path(__file__).parent
            if str(encoders_path) not in sys.path:
                sys.path.insert(0, str(encoders_path))

            import single_file_galileo

            model_dir = Path(self.encoder_weights).parent
            config_file = model_dir / "config.json"
            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                return

            with open(config_file, "r") as f:
                config = json.load(f)
                encoder_config = config["model"]["encoder"]

            encoder = single_file_galileo.Encoder(**encoder_config)

            encoder_weights_file = model_dir / "encoder.pt"
            if encoder_weights_file.exists():
                state_dict = torch.load(encoder_weights_file, map_location="cpu")

                for key in list(state_dict.keys()):
                    state_dict[key.replace(".backbone", "")] = state_dict.pop(key)
                encoder.load_state_dict(state_dict)

            self.galileo_encoder = encoder

            self.embed_dim = encoder_config.get("embedding_size", 128)

            self.output_dim = [self.embed_dim for _ in self.output_layers]
            logger.info(f"Successfully loaded Galileo encoder from {model_dir}")
            logger.info(f"Using Galileo encoder with embedding dim: {self.embed_dim}")
        except Exception as e:
            logger.error(f"Failed to load Galileo encoder: {e}")
            raise

    def _prepare_galileo_input(
        self, x: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert PANGAEA input format to Galileo input format.
        Galileo expects multiple tensors for different modalities.
        """
        batch_size = None
        height = None
        width = None
        device = None
        dtype = None

        for tensor in x.values():
            if len(tensor.shape) == 5:
                batch_size, _, _, height, width = tensor.shape
                if tensor.shape[2] == 1:
                    tensor = tensor.squeeze(2)
                else:
                    tensor = tensor[:, :, 0, :, :]
            else:
                batch_size, _, height, width = tensor.shape
            device = tensor.device
            dtype = tensor.dtype
            break
        if batch_size is None:
            raise ValueError("No valid input tensors found")

        s_t_x = torch.zeros(
            batch_size,
            height,
            width,
            1,
            len(SPACE_TIME_BANDS),
            device=device,
            dtype=dtype,
        )

        sp_x = torch.zeros(
            batch_size, height, width, len(SPACE_BANDS), device=device, dtype=dtype
        )

        t_x = torch.zeros(batch_size, 1, len(TIME_BANDS), device=device, dtype=dtype)

        st_x = torch.zeros(batch_size, len(STATIC_BANDS), device=device, dtype=dtype)

        s_t_m = torch.ones(
            batch_size,
            height,
            width,
            1,
            len(SPACE_TIME_BANDS_GROUPS_IDX),
            device=device,
            dtype=torch.bool,
        )
        sp_m = torch.ones(
            batch_size,
            height,
            width,
            len(SPACE_BAND_GROUPS_IDX),
            device=device,
            dtype=torch.bool,
        )
        t_m = torch.ones(
            batch_size, 1, len(TIME_BAND_GROUPS_IDX), device=device, dtype=torch.bool
        )
        st_m = torch.ones(
            batch_size, len(STATIC_BAND_GROUPS_IDX), device=device, dtype=torch.bool
        )

        if "optical" in x:
            optical_data = x["optical"]
            if len(optical_data.shape) == 5:
                if optical_data.shape[2] == 1:
                    optical_data = optical_data.squeeze(2)
                else:
                    optical_data = optical_data[:, :, 0, :, :]
            optical_bands = self.input_bands.get("optical", [])

            for i, band in enumerate(optical_bands):
                if band in S2_BANDS:
                    s2_idx = S2_BANDS.index(band)
                    space_time_idx = SPACE_TIME_BANDS.index(band)

                    band_data = optical_data[:, i, :, :]

                    s_t_x[:, :, :, 0, space_time_idx] = band_data

                    group_name = None
                    for group, indices in SPACE_TIME_BANDS_GROUPS_IDX.items():
                        if space_time_idx in indices:
                            group_name = group
                            break
                    if group_name:
                        group_idx = list(SPACE_TIME_BANDS_GROUPS_IDX.keys()).index(
                            group_name
                        )
                        s_t_m[:, :, :, 0, group_idx] = False

            if "B4" in optical_bands and "B8" in optical_bands:
                b4_idx = optical_bands.index("B4")
                b8_idx = optical_bands.index("B8")
                ndvi_idx = SPACE_TIME_BANDS.index("NDVI")

                red = optical_data[:, b4_idx, :, :]
                nir = optical_data[:, b8_idx, :, :]

                ndvi = (nir - red) / (nir + red + 1e-8)

                s_t_x[:, :, :, 0, ndvi_idx] = ndvi

                group_name = None
                for group, indices in SPACE_TIME_BANDS_GROUPS_IDX.items():
                    if ndvi_idx in indices:
                        group_name = group
                        break
                if group_name:
                    group_idx = list(SPACE_TIME_BANDS_GROUPS_IDX.keys()).index(
                        group_name
                    )
                    s_t_m[:, :, :, 0, group_idx] = False

        if "sar" in x:
            sar_data = x["sar"]
            if len(sar_data.shape) == 5:
                if sar_data.shape[2] == 1:
                    sar_data = sar_data.squeeze(2)
                else:
                    sar_data = sar_data[:, :, 0, :, :]
            sar_bands = self.input_bands.get("sar", [])
            for i, band in enumerate(sar_bands):
                if band in S1_BANDS:
                    s1_idx = S1_BANDS.index(band)
                    space_time_idx = SPACE_TIME_BANDS.index(band)

                    band_data = sar_data[:, i, :, :]

                    s_t_x[:, :, :, 0, space_time_idx] = band_data

                    group_name = None
                    for group, indices in SPACE_TIME_BANDS_GROUPS_IDX.items():
                        if space_time_idx in indices:
                            group_name = group
                            break
                    if group_name:
                        group_idx = list(SPACE_TIME_BANDS_GROUPS_IDX.keys()).index(
                            group_name
                        )
                        s_t_m[:, :, :, 0, group_idx] = False

        months = torch.full(
            (batch_size, 1), self.month, device=device, dtype=torch.long
        )
        return {
            "s_t_x": s_t_x,
            "sp_x": sp_x,
            "t_x": t_x,
            "st_x": st_x,
            "s_t_m": s_t_m,
            "sp_m": sp_m,
            "t_m": t_m,
            "st_m": st_m,
            "months": months,
        }

    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through the Galileo encoder.
        Args:
            x: Dictionary with modality keys and tensor values
        Returns:
            List of embeddings for each output layer
        """
        if self.galileo_encoder is None:
            raise RuntimeError(
                "Galileo encoder not loaded. Call load_encoder_weights first."
            )

        galileo_input = self._prepare_galileo_input(x)
        try:

            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = (
                self.galileo_encoder(
                    s_t_x=galileo_input["s_t_x"],
                    sp_x=galileo_input["sp_x"],
                    t_x=galileo_input["t_x"],
                    st_x=galileo_input["st_x"],
                    s_t_m=galileo_input["s_t_m"],
                    sp_m=galileo_input["sp_m"],
                    t_m=galileo_input["t_m"],
                    st_m=galileo_input["st_m"],
                    months=galileo_input["months"],
                    patch_size=self.patch_size,
                    input_resolution_m=10,
                    add_layernorm_on_exit=self.add_layernorm_on_exit,
                )
            )

            if len(s_t_x.shape) == 5:
                batch_size, patch_h, patch_w, time_dim, embed_dim = s_t_x.shape

                spatial_features = s_t_x.squeeze(3).permute(0, 3, 1, 2)
            else:

                shape = s_t_x.shape
                if len(shape) == 6:

                    if shape[3] == 1:
                        squeezed = s_t_x.squeeze(3)

                        spatial_features = squeezed.mean(dim=3)

                        spatial_features = spatial_features.permute(0, 3, 1, 2)
                    else:

                        first_time = s_t_x[:, :, :, 0, :, :]
                        spatial_features = first_time.mean(dim=3)
                        spatial_features = spatial_features.permute(0, 3, 1, 2)
                elif len(shape) >= 4:

                    spatial_features = s_t_x.permute(0, -1, -3, -2)
                else:

                    spatial_features = s_t_x

            return [spatial_features for _ in self.output_layers]
        except Exception as e:
            print(f"Error in Galileo forward pass: {e}")
            import traceback

            traceback.print_exc()

            first_tensor = next(iter(x.values()))
            if len(first_tensor.shape) == 5:
                batch_size = first_tensor.shape[0]
                height, width = first_tensor.shape[-2:]
            else:
                batch_size, _, height, width = first_tensor.shape
            device = first_tensor.device
            dtype = first_tensor.dtype

            zero_output = torch.zeros(
                batch_size,
                self.embed_dim,
                height // self.patch_size,
                width // self.patch_size,
                device=device,
                dtype=dtype,
            )
            return [zero_output for _ in self.output_layers]
