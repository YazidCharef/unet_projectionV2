import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.spatial.distance import cdist
from config import (NC_FILE, SST_FILE, TYPHOON_POSITIONS_CSV, TYPHOON_PHASES_CSV, 
                   SEQUENCE_LENGTH, CYCLOGENESIS_RADIUS, TYPHOON_RADIUS, CYCLOLYSIS_RADIUS,
                   BATCH_SIZE, TRAIN_RATIO, VAL_RATIO)

class TyphoonDataset(Dataset):
    def __init__(self, nc_file, sst_file, typhoon_positions_df, typhoon_phases_df):
        print("Loading meteorological data...")
        self.ds = xr.open_dataset(nc_file, chunks={'time': 100})
        self.sst_ds = xr.open_dataset(sst_file, chunks={'valid_time': 100})
        
        print("Synchronizing SST data...")
        self.sst_ds = self.sst_ds.rename({'valid_time': 'time'})
        self.sst_ds = self.sst_ds.sel(time=self.ds.time, method='nearest')
        
        print("Processing typhoon data...")
        self.merged_df = pd.merge(typhoon_positions_df, typhoon_phases_df, 
                                on=['Typhoon Number', 'Year'])
        
        self.time_indices = list(range(len(self.ds.time) - SEQUENCE_LENGTH + 1))
        print(f"Dataset initialized with {len(self.time_indices)} sequences")

    def __len__(self):
        return len(self.time_indices)

    def normalize_data(self, data, var_name):
        if var_name in ['u', 'v']:
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        elif var_name == 'r':
            return data / 100.0
        elif var_name == 'vo':
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        elif var_name == 'sst':
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        return data

    def __getitem__(self, idx):
        time_index = self.time_indices[idx]
        
        u = self.normalize_data(
            self.ds.u.isel(time=slice(time_index, time_index + SEQUENCE_LENGTH)).values,
            'u'
        )
        v = self.normalize_data(
            self.ds.v.isel(time=slice(time_index, time_index + SEQUENCE_LENGTH)).values,
            'v'
        )
        r = self.normalize_data(
            self.ds.r.isel(time=slice(time_index, time_index + SEQUENCE_LENGTH)).values,
            'r'
        )
        vo = self.normalize_data(
            self.ds.vo.isel(time=slice(time_index, time_index + SEQUENCE_LENGTH)).values,
            'vo'
        )
        sst = self.normalize_data(
            self.sst_ds.sst.isel(time=slice(time_index, time_index + SEQUENCE_LENGTH)).values,
            'sst'
        )

        masks = np.array([
            self.create_mask_for_time(time_index + i) 
            for i in range(SEQUENCE_LENGTH)
        ])

        X = np.stack([u, v, r, vo, sst], axis=1)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(masks, dtype=torch.float32)

        return X, y

    def create_mask_for_time(self, time_index):
        current_time = self.ds.time.isel(time=time_index).values
        mask = np.zeros((len(self.ds.latitude), len(self.ds.longitude)), dtype=bool)
        
        for _, typhoon in self.merged_df.iterrows():
            if typhoon['Cyclogenesis Start'] <= current_time <= typhoon['Cyclolysis End']:
                try:
                    positions = pd.DataFrame(eval(typhoon['Positions']))
                    positions['DateTime'] = pd.to_datetime(positions['DateTime'])
                    closest_pos = positions.loc[(positions['DateTime'] - current_time).abs().idxmin()]
                    
                    if typhoon['Cyclogenesis Start'] <= current_time < typhoon['Typhoon Start']:
                        radius = CYCLOGENESIS_RADIUS
                    elif typhoon['Typhoon Start'] <= current_time < typhoon['Cyclolysis Start']:
                        radius = TYPHOON_RADIUS
                    else:
                        radius = CYCLOLYSIS_RADIUS

                    lons, lats = np.meshgrid(self.ds.longitude, self.ds.latitude)
                    coords = np.dstack((lats.ravel(), lons.ravel()))[0]
                    center = np.array([[float(closest_pos['Latitude']), 
                                      float(closest_pos['Longitude'])]])
                    distances = cdist(coords, center).reshape(lats.shape)
                    mask = np.logical_or(mask, distances <= (radius / 111))
                    
                except Exception as e:
                    print(f"Error processing typhoon at time {current_time}: {e}")
                    continue

        return mask

def get_data_loaders(batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO, 
                    val_ratio=VAL_RATIO, data_fraction=1.0):
    print("\nInitializing data loaders...")
    print(f"Using batch size: {batch_size}")

    typhoon_positions_df = pd.read_csv(
        TYPHOON_POSITIONS_CSV, 
        parse_dates=['Birth', 'Death (Latest)', 'Data Start', 'Data End']
    )
    typhoon_phases_df = pd.read_csv(
        TYPHOON_PHASES_CSV, 
        parse_dates=['Cyclogenesis Start', 'Cyclogenesis End', 
                    'Typhoon Start', 'Typhoon End', 
                    'Cyclolysis Start', 'Cyclolysis End']
    )
    
    full_dataset = TyphoonDataset(NC_FILE, SST_FILE, typhoon_positions_df, typhoon_phases_df)
    
    if data_fraction < 1.0:
        num_samples = int(len(full_dataset) * data_fraction)
        indices = torch.randperm(len(full_dataset))[:num_samples]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nSplitting dataset:")
    print(f"- Train: {train_size} samples")
    print(f"- Validation: {val_size} samples")
    print(f"- Test: {test_size} samples")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    print("\nData loaders created successfully")
    return train_loader, val_loader, test_loader