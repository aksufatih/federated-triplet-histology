import argparse
import csv
import os

import nrrd
import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import zoom
import SimpleITK as sitk


class Processor:

    def __init__(self, df, datapath, outpath, target_spacing=(0.977, 0.977)):
        self.df = df
        self.datapath = datapath
        self.outpath = outpath
        self.target_spacing = tuple(float(t) for t in target_spacing)

    def __call__(self):
        with open(os.path.join(self.outpath, 'center_info.csv'), mode='a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['x_min_max', 'y_min_max', 'center', 'shape', 'X', 'area'])
            for pid in self.df['Subject ID'].unique():
                if pid == 'LUNG1-128':  # Segmentation problem
                    continue
                if os.path.exists(os.path.join(self.outpath, f'{pid}_1.nrrd')):
                    continue

                print("Processing", pid)

                img, origin, spacing = self.read_img(os.path.join(self.datapath, self.df[(self.df['Subject ID'] == pid)
                                                                                         & (self.df['Modality'] == 'CT')][
                    'File Location'].item()))

                mask = self.read_seg(os.path.join(self.datapath,
                                                  self.df[(self.df['Subject ID'] == pid)
                                                          & (self.df['Modality'] == 'RTSTRUCT')]['File Location'].item(),
                                                  '1-1.dcm'), img.shape, origin, spacing)

                img_resampled = zoom(img, (1,
                                           spacing[0] / self.target_spacing[0],
                                           spacing[1] / self.target_spacing[1]), order=1)

                mask_resampled = zoom(mask, (1,
                                             spacing[0] / self.target_spacing[0],
                                             spacing[1] / self.target_spacing[1]), order=0)

                nonzeros = np.nonzero(mask_resampled)

                for slice_no, i in enumerate(set(nonzeros[0].tolist())):
                    x_min = nonzeros[2][nonzeros[0] == i].min()
                    x_max = nonzeros[2][nonzeros[0] == i].max()
                    y_min = nonzeros[1][nonzeros[0] == i].min()
                    y_max = nonzeros[1][nonzeros[0] == i].max()

                    # To write the whole slice and store the information about the tumor
                    csvwriter.writerow([(x_min, x_max), (y_min, y_max), (int(x_min + (x_max - x_min)/2), int(y_min + (y_max - y_min)/2)), (mask_resampled.shape[1], mask_resampled.shape[2]), f'{pid}_{slice_no+1}.nrrd', (x_max-x_min)*(y_max-y_min)])
                    nrrd.write(os.path.join(self.outpath, f'{pid}_{slice_no+1}.nrrd'), img_resampled[i])

    def read_img(self, path):

        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        img_array = sitk.GetArrayFromImage(image)

        photometric_interpretation = reader.GetMetaData(slice=1, key='0028|0004').strip()
        if photometric_interpretation == 'MONOCHROME1':
            img_array = img_array * -1 + np.max(img_array)

        return img_array, origin, spacing

    def read_seg(self, path, shape, origin, spacing):

        mask_array = np.zeros(shape, dtype=np.int8)
        contour = pydicom.read_file(path, force=True)
        for roi in contour.StructureSetROISequence:
            if roi.ROIName == 'GTV-1':
                ref = roi.ROINumber

        for roi in contour.ROIContourSequence:
            if roi.ReferencedROINumber == ref:
                for c in roi.ContourSequence:
                    coordinates = c.ContourData
                    for i in range(0, len(coordinates), 3):
                        x1 = coordinates[i]
                        y1 = coordinates[i + 1]
                        z1 = coordinates[i + 2]
                        # Convert the coordinates into pixel data
                        mask_array[round((z1 - origin[2]) / spacing[2])][round((y1 - origin[1]) / spacing[1])][
                            round((x1 - origin[0]) / spacing[0])] = 1

        return mask_array


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--metadata_path', required=True, help='Path to metadata file')
    parser.add_argument('-dp', '--data_path', required=True, help='Path to data files')
    parser.add_argument('-op', '--out_path', required=True, help='Path to folder to store the processed data')
    parser.add_argument('-ts', '--target_spacing', nargs='+', help='Target spacing')
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_path)
    if args.target_spacing is not None:
        processor = Processor(df, args.data_path, args.out_path, args.target_spacing)
    else:
        processor = Processor(df, args.data_path, args.out_path)
    processor()
