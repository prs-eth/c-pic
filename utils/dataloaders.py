import os
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GroupKFold, KFold
import h5py as h5
import os.path as osp
import scipy as sp
import torch
import pydicom
import nibabel as nib
import pickle
import glob
import pandas as pd
import numpy as np
from typing import Optional, Iterable


pydicom.config.image_handlers = ['gdcm_handler']


class Dicom(Dataset):
    def __init__(
            self,
            path: str,
            train_patients: Iterable[str],
            features: Optional[torch.Tensor] = None,
            resolution: Optional[Iterable[int]] = None,
            part: str = 'train',
            cache: bool = True):
        self.part = pd.read_csv(osp.join(path, part + '.csv'))
        self.part = pd.merge(
            self.part,
            self.part.groupby('Patient').nth(0)['Weeks'].rename('Weeks_init'), on='Patient', how='left')
        self.part = pd.merge(
            self.part,
            self.part.groupby('Patient').nth(0)['FVC'].rename('FVC_init'), on='Patient', how='left')
        self.part = pd.merge(
            self.part,
            self.part.groupby('Patient').nth(0)['Percent'].rename('Percent_init'), on='Patient', how='left')
        self.part['Weeks_norm'] = self.part['Weeks'] - self.part['Weeks_init']
        self.part['FVC_norm'] = self.part['FVC'] / self.part['FVC_init']
        self.part['dcm_path'] = path + '/' + part + '/' + self.part.Patient
        self.part['Group'] = np.zeros(len(self.part), dtype=np.int64)
        self.cache = cache

        for i, p in enumerate(self.part.Patient.unique()):
            idx = (self.part.Patient == p)
            self.part.Group[idx] = np.ones(idx.sum(), dtype=np.int64) * i

        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        self.enc_sex.fit(self.part.Sex)
        smoking_status = self.enc_smok.fit_transform(self.part.SmokingStatus)
        self.onehotenc_smok.fit(smoking_status.reshape(-1, 1))
        self.fvc_max = self.part.FVC.max()
        self.fvc_min = self.part.FVC.min()
        self.features = features
        self.resolution = resolution or [30, 512, 512]
        print('DICOM dataset created with resolution {}'.format(self.resolution))

        self.X_train = self.part[self.part.Patient.isin(train_patients)]
        self.train_patients = train_patients

        self.percent = (self.X_train.Percent_init.max(), self.X_train.Percent_init.min())
        self.fvc = (self.X_train.FVC_init.max(), self.X_train.FVC_init.min())
        self.weeks = (self.X_train.Weeks_norm.max(), self.X_train.Weeks_norm.min())
        self.age = (self.X_train.Age.max(), self.X_train.Age.min())

    def standardisation(self, x, u, s):
        return (x - u) / s

    def normalization(self, x, ma, mi):
        return (x - mi) / (ma - mi)

    def load_scans(self, dcm_path):
        # in this competition we have missing values in ImagePosition, this is why we are sorting by filename number
        files = os.listdir(dcm_path)
        file_nums = [np.int(file.split('.')[0]) for file in files]
        sorted_file_nums = np.sort(file_nums)[::-1]
        slices = [pydicom.dcmread(dcm_path + '/' + str(file_num) + '.dcm') for file_num in sorted_file_nums]

        return slices

    def set_outside_scanner_to_air(self, raw_pixelarrays):
        # in OSIC we find outside-scanner-regions with raw-values of -2000. 
        # Let's threshold between air (0) and this default (-2000) using -1000
        raw_pixelarrays[raw_pixelarrays <= -1000] = 0
        return raw_pixelarrays

    def transform_to_hu(self, slices):
        images = np.stack([file.pixel_array for file in slices])
        images = images.astype(np.int16)

        images = self.set_outside_scanner_to_air(images)

        # convert to HU
        for n in range(len(slices)):

            intercept = slices[n].RescaleIntercept
            slope = slices[n].RescaleSlope

            if slope != 1:
                images[n] = slope * images[n].astype(np.float64)
                images[n] = images[n].astype(np.int16)

            images[n] += np.int16(intercept)

        return np.array(images, dtype=np.int16)

    def resample(self, image, target_shape):
        return sp.ndimage.interpolation.zoom(image, np.array(target_shape) / image.shape, mode='nearest')

    def __len__(self):
        return len(self.part)

    def __getitem__(self, idx):
        if self.features is None:
            if self.cache:
                suffix = 'normalized'
                fname = self.part.dcm_path.values[idx] + '_prep_{}'.format(suffix)
                if os.path.exists(fname):
                    with open(fname, 'rb') as f:
                        scan_resampled = pickle.load(f)
                else:
                    scan = self.load_scans(self.part.dcm_path.values[idx])
                    scan_hu = self.transform_to_hu(scan)
                    scan_resampled = self.resample(scan_hu, self.resolution).astype(np.float32)
                    with open(fname, 'wb') as f:
                        pickle.dump(scan_resampled, f)
            else:
                scan = self.load_scans(self.part.dcm_path.values[idx])
                scan_hu = self.transform_to_hu(scan)
                scan_resampled = self.resample(scan_hu, self.resolution).astype(np.float32)
            sample = ((scan_resampled - scan_resampled.min()) / (scan_resampled.max() - scan_resampled.min()))
            # sample = scan_resampled
            sample = torch.tensor(sample).float()
            old_shape = torch.tensor(sample.shape)
            new_shape = 2 ** torch.ceil(torch.log(old_shape.float()) / np.log(2)).long()

            scan_features = torch.nn.functional.pad(
                sample,
                tuple(torch.cat([
                    torch.zeros(len(old_shape), dtype=torch.int64),
                    torch.flip(new_shape - old_shape, (0,))]).numpy()))
        else:
            scan_features = self.features[idx].reshape(-1)

        sex = self.enc_sex.transform([self.part.iloc[idx].Sex])
        smoking = self.onehotenc_smok.transform(
            self.enc_smok.transform([self.part.iloc[idx].SmokingStatus]).reshape(-1, 1))

        features = [
             self.normalization(np.array(self.part.iloc[idx].Weeks_norm), self.weeks[0], self.weeks[1]),
             self.normalization(np.array(self.part.iloc[idx].Percent), self.percent[0], self.percent[1]),
             self.normalization(np.array(self.part.iloc[idx].FVC_init), self.fvc[0], self.fvc[1]),
             self.normalization(np.array(self.part.iloc[idx].Age), self.age[0], self.age[1])]

        features = torch.tensor(
             features + list(np.asarray(smoking.todense())[0]) + [sex[0]]).float()
        target = self.part.iloc[idx].FVC_norm

        return scan_features, features, target, torch.tensor(self.part.iloc[idx].FVC_init)

    def group_kfold(self, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(self.part, groups=self.part.Group.values):
            train = Subset(self, train_idx)
            val = Subset(self, val_idx)
            yield train, val


class BRATS2020(Dataset):
    def __init__(self, root: str):
        self.root = root

        self.meta = pd.read_csv(osp.join(root, 'survival_info.csv'))
        # NOTE(aelphy): remove one patient with strange target
        self.meta = self.meta[self.meta.Survival_days != 'ALIVE (361 days later)']
        self.patients = list(self.meta.Brats20ID)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        t1_scan = np.array(
            nib.load(
                osp.join(
                    self.root,
                    self.patients[idx],
                    self.patients[idx] + '_t1.nii')).get_fdata())

        target_shape = [256, 256, 256]
        t1_scan = sp.ndimage.interpolation.zoom(
            t1_scan, np.array(target_shape) / t1_scan.shape,
            mode='nearest')
        t1_scan = torch.tensor(t1_scan)
        y = self.meta[self.meta.Brats20ID == self.patients[idx]].Survival_days
        # NOTE(aelphy): 5 years normilisation
        y = torch.tensor(np.array(pd.to_numeric(y)) / 365 / 5)

        return t1_scan.double(), y.double()

    def kfold(self, n_splits: int):
        kf = KFold(n_splits=n_splits)
        for train_idx, val_idx in kf.split(np.arange(len(self))):
            train = Subset(self, list(train_idx))
            val = Subset(self, list(val_idx))
            yield train, val
