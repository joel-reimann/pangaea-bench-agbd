import torch
import rasterio as rs
import numpy as np
import os
import argparse
import importlib
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from inference_helper import process_S2_tile
from inference_ds import InferenceDataset_v3, get_patch_weight
from pangaea.engine.data_preprocessor import NormalizeMeanStd
from pangaea.decoders.upernet import RegUPerNet


def get_encoder_class_and_params(model_name: str, agbd_cfg: DictConfig):
    encoder_path = f"/scratch2/reimannj/pangaea-bench/configs/encoder/{model_name}.yaml"
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder config not found at: {encoder_path}")
    encoder_cfg = OmegaConf.load(encoder_path)
    params = OmegaConf.to_container(encoder_cfg, resolve=False)
    if (
        "input_bands" in params
        and isinstance(params["input_bands"], str)
        and "${" in params["input_bands"]
    ):
        params["input_bands"] = {
            "optical": OmegaConf.to_container(agbd_cfg.bands.optical)
        }
    if (
        "input_size" in params
        and isinstance(params["input_size"], str)
        and "${" in params["input_size"]
    ):
        params["input_size"] = agbd_cfg.img_size
    target_class_str = params.pop("_target_")
    module_name, class_name = target_class_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    encoder_class = getattr(module, class_name)
    return encoder_class, params


def predict_tile(decoder, dataset, full_shape, device):
    """
    Final, corrected prediction function implementing the "crop-and-place" method.
    This avoids blending and should eliminate all tiling and smearing artifacts.
    """
    full_pred = np.zeros(full_shape, dtype=np.float32)
    crop = dataset.pred_crop

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Predicting Patches"):

            patch_data = dataset[i][0]

            patch_tensor = (
                torch.from_numpy(patch_data)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .float()
                .to(device)
            )
            model_input = {"optical": patch_tensor}

            pred = decoder(model_input).squeeze().cpu().numpy()

            cropped_pred = pred[crop:-crop, crop:-crop]

            y1, y2, x1, x2 = dataset.images_indices[i]

            h, w = y2 - y1, x2 - x1

            full_pred[y1:y2, x1:x2] = cropped_pred[:h, :w]

    return full_pred


def run_inference():
    parser = argparse.ArgumentParser(description="Inference for AGBD dataset")
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Name of the encoder config file (without .yaml)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output GeoTIFF",
    )
    args = parser.parse_args()

    agbd_cfg = OmegaConf.load(
        "/scratch2/reimannj/pangaea-bench/configs/dataset/agbd.yaml"
    )
    s2_product = "S2B_MSIL2A_20200324T135109_N9999_R024_T21JWM_20240422T081051"
    s2_prod_path = "/scratch2/reimannj/AGBD/Models/"

    transform, upsampling_shape, s2_bands, crs, _, _, _, _, _, _, meta = (
        process_S2_tile(s2_product, s2_prod_path)
    )

    s2_bands.pop("SCL")
    s2_bands = {band: (val - 1000) / 10000 for band, val in s2_bands.items()}
    s2_order_pangaea = agbd_cfg.bands["optical"]
    s2_order_data = [
        b.replace("B", "B0") if len(b) == 2 and b[1].isdigit() else b
        for b in s2_order_pangaea
    ]
    s2_bands_array = np.moveaxis(
        np.array([s2_bands[band] for band in s2_order_data]), 0, -1
    )

    optical_preprocessor = NormalizeMeanStd(
        data_mean={"optical": torch.tensor(agbd_cfg.data_mean["optical"])},
        data_std={"optical": torch.tensor(agbd_cfg.data_std["optical"])},
    )
    s2_bands_tensor = (
        torch.from_numpy(s2_bands_array).permute(2, 0, 1).float().unsqueeze(1)
    )
    normalized_s2_bands = optical_preprocessor({"image": {"optical": s2_bands_tensor}})[
        "image"
    ]["optical"]
    image = normalized_s2_bands.squeeze(1).permute(1, 2, 0).numpy()

    encoder_class, encoder_params = get_encoder_class_and_params(args.encoder, agbd_cfg)
    encoder = encoder_class(**encoder_params)
    decoder = RegUPerNet(encoder=encoder, finetune=False, channels=512)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        decoder.load_state_dict(ckpt["model"])
    else:
        print(f"ERROR: Checkpoint file not found at {args.checkpoint}.")
        return

    decoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)

    patch_size = (
        encoder_params.get("input_size", 224),
        encoder_params.get("input_size", 224),
    )

    pred_crop = np.array([64, 64, 64, 64])

    inference_dataset = InferenceDataset_v3(
        img=image, patch_size=patch_size, pred_crop=pred_crop, cfg={}
    )

    predictions = predict_tile(decoder, inference_dataset, upsampling_shape, device)

    predictions[predictions < 0] = 0
    predictions[predictions > 65535] = 65535
    predictions = predictions.astype(np.uint16)

    output_filename = os.path.join(args.output_dir, f"agbd_prediction_{s2_product}.tif")
    meta.update(
        driver="GTiff",
        dtype=np.uint16,
        count=1,
        compress="lzw",
        nodata=65535,
        transform=transform,
    )
    with rs.open(output_filename, "w", **meta) as dst:
        dst.write(predictions, 1)
        dst.set_band_description(1, "AGBD")

    print(f"Successfully saved prediction to {output_filename}")


if __name__ == "__main__":
    run_inference()
    print("Inference done!")
