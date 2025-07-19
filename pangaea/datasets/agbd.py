"""
AGBD Dataset for PANGAEA, task: Regression (Biomass Estimation)

Adapted from the official AGBD GEDI Dataset implementation (see https://github.com/ghjuliasialelli/AGBD/blob/d8e7287bbe787277d64f25806f3c991b9cec7076/Models/dataset.py#L32)
This version is designed to be as close as possible to the proven AGBD code, with only minimal changes for PANGAEA config-driven initialization and output conventions.

- All band order, and nodata logic is preserved.
- Handles multiple HDF5 files and years, using split files and mapping.
- Patch extraction and stacking for all modalities.
- Label extraction for regression.

DO NOT MODIFY unless you have read both the AGBD and PANGAEA documentation and understand the integration risks.
"""
import h5py
import torch
from pangaea.datasets.base import RawGeoFMDataset
import numpy as np
import os
from os.path import join, exists
import pickle


# --- Begin copied helper functions from AGBD/Models/dataset.py ---
# Define the nodata values for each data source (from original AGBD)
NODATAVALS = {'S2_bands' : 0, 'CH': 255, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}

def initialize_index(fnames, mode, chunk_size, path_mapping, path_h5) :
    """
    This function creates the index for the dataset. The index is a dictionary which maps the file
    names (`fnames`) to the tiles that are in the `mode` (train, val, test); and the tiles to the
    number of chunks that make it up.
    Args:
    - fnames (list): list of file names
    - mode (str): the mode of the dataset (train, val, test)
    - chunk_size (int): the size of the chunks
    - path_mapping (str): the path to the file mapping each mode to its tiles
    - path_h5 (str): the path to the HDF5 files
    Returns:
    - idx (dict): dictionary mapping the file names to the tiles and the tiles to the chunks
    - total_length (int): the total number of chunks in the dataset
    """

    # Load the mapping from mode to tile name
    with open(join(path_mapping, 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)

    # Iterate over all files
    idx = {}
    for fname in fnames :
        idx[fname] = {}
        with h5py.File(join(path_h5, fname), 'r') as f:
            # Get the tiles in this file which belong to the mode
            all_tiles = list(f.keys())
            tiles = np.intersect1d(all_tiles, tile_mapping[mode])
            # Iterate over the tiles
            for tile in tiles :
                # Get the number of patches in the tile
                n_patches = len(f[tile]['GEDI']['agbd'])
                idx[fname][tile] = n_patches // chunk_size
    total_length = sum(sum(v for v in d.values()) for d in idx.values())
    return idx, total_length

# ===============================================================

def find_index_for_chunk(index, n, total_length):
    """
    For a given `index` and `n`-th chunk, find the file, tile, and row index corresponding
    to this chunk.
    Args:
    - index (dict): dictionary mapping the files to the tiles and the tiles to the chunks
    - n (int): the n-th chunk
    Returns:
    - file_name (str): the name of the file
    - tile_name (str): the name of the tile
    - chunk_within_tile (int): the chunk index within the tile
    """

    # Check that the chunk index is within bounds
    assert n < total_length, "The chunk index is out of bounds"
    # Iterate over the index to find the file, tile, and row index
    cumulative_sum = 0
    for file_name, file_data in index.items():
        for tile_name, num_rows in file_data.items():
            if cumulative_sum + num_rows > n:
                # Calculate the row index within the tile
                chunk_within_tile = n - cumulative_sum
                return file_name, tile_name, chunk_within_tile
            cumulative_sum += num_rows

# --- End copied helper functions ---

class AGBD(RawGeoFMDataset):
    """
    PANGAEA-compatible dataset for AGBD, adapted from GEDIDataset.
    Uses only biomes_splits_to_name.pkl for split logic, matching the original AGBD code.
    Reads config fields as in agbd.yaml, builds index from tile mapping, and loads patches from multiple HDF5 files.
    """
    def __init__(
            self, 
            dataset_name: str, 
            root_path: str, 
            hdf5_dir: str, 
            mapping_path: str, 
            norm_path: str, 
            version: int, 
            split: str, 
            chunk_size: int, 
            years: set, 
            img_size: int, 
            multi_modal: bool, 
            multi_temporal: int, 
            classes: list, 
            num_classes: int, 
            ignore_index: int, 
            bands: dict[str, list[str]], 
            distribution: list[int], 
            data_mean: dict[str, list[str]], 
            data_std: dict[str, list[str]], 
            data_min: dict[str, list[str]], 
            data_max: dict[str, list[str]], 
            download_url: str, 
            auto_download: bool,
            debug:bool,
    ):
        super(AGBD, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )
        # SOURCE OF CODE IS dataset.py https://github.com/ghjuliasialelli/AGBD/blob/main/Models/dataset.py, CHANGES MADE FOR PANGAEA COMPATIBILITY

        # Get the paths
        self.root_path = root_path
        self.h5_path = hdf5_dir
        self.norm_path = norm_path
        self.mapping = mapping_path

        # Get the parameters
        self.dataset_name = dataset_name        
        self.mode = split
        self.split = split
        self.chunk_size = chunk_size                                             
        self.years = years
        self.img_size = img_size
        self.patch_size = (img_size, img_size)
        
        # Check that the mode is valid
        assert self.mode in ['train', 'val', 'test'], "The mode must be one of 'train', 'val', 'test'."

        # define variable for each available feature group
        self.bands = bands
        self.S2_bands_optical = bands['optical']  # S2 optical bands
        self.alos_bands = bands['sar']  # SAR bands

        # Get the file names, if debug then only one file to allow faster computation
        self.debug=debug
        self.fnames = []
        if self.debug:
            self.fnames += [f'data_subset-2019-v{version}_16-20.h5']
        else:
            for year in self.years:
                self.fnames += [f'data_subset-{year}-v{version}_{i}-20.h5' for i in range(20)]

        # Initialize the index
        self.index, self.length = initialize_index(self.fnames, self.mode, self.chunk_size, self.mapping, self.h5_path)

        # Aggressive debug mode: drastically reduce number of samples for fast debug runs
        if self.debug:
            self.length = min(self.length, 16)  # Use only 16 samples for both train and val

        # NOTE: Don't open all file handles at once - this can cause memory issues and conflicts
        # We'll open them on-demand in __getitem__ instead
        self.handles = {}

        # Define the window size and patch handling
        assert self.patch_size[0] == self.patch_size[1], "The patch size must be square"
        
        # AGBD data is stored as 25x25 patches in HDF5 files as per the original paper
        self.original_patch_size = 25
        self.center = 12  # Center of 25x25 patch
        self.window_size = self.original_patch_size // 2  # 12
        
        # The dataset should always return 25x25 patches as per AGBD paper
        # PANGAEA's preprocessing pipeline will handle any size adjustments for different encoders
        if self.img_size != 25:
            print(f"[AGBD] INFO: img_size={self.img_size} specified for ViT alignment.")
            print(f"[AGBD] Dataset returns 25x25 patches, preprocessing pads to {self.img_size}x{self.img_size}.")
            print(f"[AGBD] This fixes ViT token alignment (25 % 16 != 0 causes checkerboard artifacts).")

        # For PANGAEA compatibility: evaluator expects further attribute (see biomassters.py etc.)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.classes = classes
        self.distribution = distribution
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.download_url = download_url
        self.auto_download = auto_download

    @staticmethod
    def download(self, silent=False):
        # Pass since data needs to be downloaded manually, see download.sh in https://github.com/ghjuliasialelli/AGBD/tree/d8e7287bbe787277d64f25806f3c991b9cec7076/Models
        pass

    def __len__(self):
        # source code is dataset.py https://github.com/ghjuliasialelli/AGBD/blob/main/Models/dataset.py
        return self.length

    def __getitem__(self, n):
        """Returns the i-th item of the dataset.
       Args:
           i (int): index of the item
       Raises:
           NotImplementedError: raise if the method is not implemented
       Returns:
           dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
           {"image":
               {"optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),},
           "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
           regression datasets.,
            "metadata": dict}.
       """
        # source code is dataset.py https://github.com/ghjuliasialelli/AGBD/blob/main/Models/dataset.py, CHANGES MADE FOR PANGAEA COMPATIBILITY

        # CRITICAL DEBUG: Track every step of data loading
        # print(f"\n[AGBD DEBUG] __getitem__ called with n={n}")
        # print(f"[AGBD DEBUG] Dataset length: {self.length}")
        # print(f"[AGBD DEBUG] Patch size: {self.patch_size}")
        # print(f"[AGBD DEBUG] Center: {self.center}, Window size: {self.window_size}")
        # print(f"[AGBD DEBUG] Available bands: optical={self.bands.get('optical', 'None')}, sar={self.bands.get('sar', 'None')}")

        # Find the file, tile, and row index corresponding to this chunk
        file_name, tile_name, idx = find_index_for_chunk(self.index, n, self.length)
        # print(f"[AGBD DEBUG] Loading from file: {file_name}, tile: {tile_name}, idx: {idx}")
        
        # Open file handle on-demand for better memory management
        if file_name not in self.handles:
            self.handles[file_name] = h5py.File(join(self.h5_path, file_name), 'r')
        f = self.handles[file_name]

        # Set the order and indices for the Sentinel-2 bands

        # --- PATCH: Naming convention of bands not everywhere equal - hence make list for the loop in following lines
        bands_name = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        # --- PATCH END ---

        if not hasattr(self, 's2_order') : self.s2_order = list(f[tile_name]['S2_bands'].attrs['order'])
        if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in bands_name]

        # Set the order for the ALOS bands
        if "sar" in self.bands.keys():
            if not hasattr(self, 'alos_order'):
                self.alos_order = f[tile_name]['ALOS_bands'].attrs['order']
                # print(f"[AGBD DEBUG] ALOS band order from file: {self.alos_order}")
                # print(f"[AGBD DEBUG] ALOS bands requested: {self.alos_bands}")
        else:
            # print(f"[AGBD DEBUG] No SAR bands in configuration")
            pass

        
        # Sentinel-2 bands
        if "optical" in self.bands.keys():
            
            # print(f"[AGBD DEBUG] Loading optical bands...")
            
            # Get the bands
            s2_bands = f[tile_name]['S2_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)
            # print(f"[AGBD DEBUG] Raw S2 bands shape: {s2_bands.shape}")
            # print(f"[AGBD DEBUG] Raw S2 bands range: [{s2_bands.min():.6f}, {s2_bands.max():.6f}]")
            # print(f"[AGBD DEBUG] Raw S2 bands mean: {s2_bands.mean():.6f}")
            # print(f"[AGBD DEBUG] Raw S2 bands non-zero count: {np.count_nonzero(s2_bands)}/{s2_bands.size}")

            # Get the BOA offset, if it exists
            if 'S2_boa_offset' in f[tile_name]['Sentinel_metadata'].keys() : 
                s2_boa_offset = f[tile_name]['Sentinel_metadata']['S2_boa_offset'][idx]
                # print(f"[AGBD DEBUG] BOA offset: {s2_boa_offset}")
            else: 
                s2_boa_offset = 0
                # print(f"[AGBD DEBUG] No BOA offset found, using 0")

            # [PATCH: S2_BANDS/BOA_OFFSET DTYPE SAFETY + DIAGNOSTIC]
            s2_bands = s2_bands.astype(np.float32)
            s2_boa_offset = np.array(s2_boa_offset).astype(np.float32)
            # [END PATCH]

            # Get the surface reflectance values
            sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
            sr_bands[s2_bands == 0] = 0
            sr_bands[sr_bands < 0] = 0
            s2_bands = sr_bands
            
            # print(f"[AGBD DEBUG] After surface reflectance conversion:")
            # print(f"[AGBD DEBUG] SR bands range: [{s2_bands.min():.6f}, {s2_bands.max():.6f}]")
            # print(f"[AGBD DEBUG] SR bands mean: {s2_bands.mean():.6f}")
            # print(f"[AGBD DEBUG] SR bands non-zero count: {np.count_nonzero(s2_bands)}/{s2_bands.size}")

            # define nodata values (changed from original code since normalization separately)
            s2_bands = np.where(s2_bands == NODATAVALS['S2_bands'], 0, s2_bands)

            # Return raw surface reflectance values - PANGAEA preprocessing will handle normalization
            # print(f"[AGBD DEBUG] Using raw surface reflectance values")
            # print(f"[AGBD DEBUG] Surface reflectance range: [{s2_bands.min():.6f}, {s2_bands.max():.6f}]")

            s2_bands = s2_bands[:, :, self.s2_indices]
            # print(f"[AGBD DEBUG] After band selection - S2 shape: {s2_bands.shape}")
            # print(f"[AGBD DEBUG] Selected band indices: {self.s2_indices}")
            # print(f"[AGBD DEBUG] Band order: {getattr(self, 's2_order', 'Not set')}")
        else:
            # print(f"[AGBD DEBUG] No optical bands requested")
            s2_bands = None            
        
        if "sar" in self.bands.keys():
            # print(f"[AGBD DEBUG] Loading SAR bands...")
            
            # Get the bands
            alos_bands = f[tile_name]['ALOS_bands'][idx, self.center - self.window_size : self.center + self.window_size + 1, self.center - self.window_size : self.center + self.window_size + 1, :].astype(np.float32)
            # print(f"[AGBD DEBUG] Raw ALOS bands shape: {alos_bands.shape}")
            # print(f"[AGBD DEBUG] Raw ALOS bands range: [{alos_bands.min():.6f}, {alos_bands.max():.6f}]")
            # print(f"[AGBD DEBUG] Raw ALOS bands mean: {alos_bands.mean():.6f}")
            # print(f"[AGBD DEBUG] Raw ALOS bands non-zero count: {np.count_nonzero(alos_bands)}/{alos_bands.size}")

            # Get the gamma naught values
            # Mask zeros and negatives before log10 to avoid divide by zero (caused crash)
            alos_bands = np.where(alos_bands <= 0, 1e-6, alos_bands)
            alos_bands = np.where(alos_bands == NODATAVALS['ALOS_bands'], -9999.0, 10 * np.log10(np.power(alos_bands, 2)) - 83.0)
            
            # Return raw gamma naught values - PANGAEA preprocessing will handle normalization
            # print(f"[AGBD DEBUG] Using raw SAR gamma naught values")
            # print(f"[AGBD DEBUG] SAR gamma naught range: [{alos_bands.min():.6f}, {alos_bands.max():.6f}]")
            
            # print(f"[AGBD DEBUG] ALOS bands non-zero count: {np.count_nonzero(alos_bands)}/{alos_bands.size}")
        else:
            # print(f"[AGBD DEBUG] No SAR bands requested")
            alos_bands = None


        # Get the GEDI target data and turn into tensor
        agbd = f[tile_name]['GEDI']['agbd'][idx]
        # print(f"[AGBD DEBUG] Raw AGBD value: {agbd}")
        # print(f"[AGBD DEBUG] AGBD type: {type(agbd)}")
        
        # Return raw biomass values - PANGAEA preprocessing will handle normalization
        # print(f"[AGBD DEBUG] Using raw AGBD biomass values")
        
        agbd = torch.from_numpy(np.array(agbd, dtype = np.float32)).to(torch.float)
        # print(f"[AGBD DEBUG] AGBD tensor value: {agbd.item()}")

        # ---------- FOR PANGAEA COMPATIBILITY ---------------------------------------
        # PANGAEA RegEvaluator expects target shape to match image shape (H x W)
        # It computes loss using center pixel: logits[:,pxl,pxl] vs target[:,pxl,pxl]
        # Create 25x25 patch tensor with AGBD value - center pixel will be used for loss
        target = torch.full((25, 25), float(agbd), dtype=torch.float32)
        print(f"[AGBD DEBUG] Created target tensor shape: {target.shape}")
        print(f"[AGBD DEBUG] Target tensor center value: {target[12, 12].item()}")
        print(f"[AGBD DEBUG] AGBD biomass value: {agbd.item():.2f} Mg/ha")
        
        # define image and target as PANGAEA expects it (images' shape =  C T H W)
        image = {}
        
        if "optical" in self.bands.keys() and s2_bands is not None:
            image['optical'] = torch.from_numpy(s2_bands).permute(2, 0, 1).unsqueeze(1).float()
            print(f"[AGBD DEBUG] Optical tensor shape: {image['optical'].shape}")
            print(f"[AGBD DEBUG] Expected: (C=12, T=1, H=25, W=25)")
            print(f"[AGBD DEBUG] Optical tensor range: [{image['optical'].min().item():.6f}, {image['optical'].max().item():.6f}]")
            # print(f"[AGBD DEBUG] Optical tensor shape: {image['optical'].shape}")
            # print(f"[AGBD DEBUG] Optical tensor range: [{image['optical'].min().item():.6f}, {image['optical'].max().item():.6f}]")
        
        # Add SAR to image dictionary if available (simplified from working version)
        if "sar" in self.bands.keys() and alos_bands is not None:
            image['sar'] = torch.from_numpy(alos_bands).permute(2, 0, 1).unsqueeze(1).float()
            # print(f"[AGBD DEBUG] SAR tensor shape: {image['sar'].shape}")
            # print(f"[AGBD DEBUG] SAR tensor range: [{image['sar'].min().item():.6f}, {image['sar'].max().item():.6f}]")
        
        # print(f"[AGBD DEBUG] Final image keys: {list(image.keys())}")
        # print(f"[AGBD DEBUG] Final target shape: {target.shape}, value at center: {target[12, 12].item()}")
        
        # return dictionary with expected structure
        return {'image': image, 'target': target, 'metadata': {}}

    def __del__(self):
        """Clean up HDF5 file handles"""
        if hasattr(self, 'handles'):
            for handle in self.handles.values():
                try:
                    handle.close()
                except:
                    pass  # Handle might already be closed

    # Define the nodata values for each data source (from original AGBD)
NODATAVALS = {'S2_bands' : 0, 'CH': 255, 'ALOS_bands': 0, 'DEM': -9999, 'LC': 255}

