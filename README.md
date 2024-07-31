# Enhancing NSCLC Histological Subtype Classification: A Federated Learning Approach Using Triplet Loss

## Usage

### Preprocessing
Preprocess the original images

```bash
python preprocess.py -mp /path/to/metadata -dp /path/to/original/data -op /path/to/output/folder -ts target_spacing
```

`/path/to/metadata` is the path to a CSV file with at least three columns 'Subject ID', 'Modality', and 'File Location'. 
'File Location' should be relative to `/path/to/original/data`. 'Modality' should be 'CT' for the images and 'RTSTRUCT' for segmentation.

`target_spacing` should be a tuple of floats, default is `(0.977, 0.977)`.

The preprocessed files will be saved with the name `<Subject ID>_<slice_idx>.nrrd` to the `/path/to/output/folder`.
Additionally, another file will be written to the same output folder with the name `center_info.csv` containing the center points of tumor bounding boxes.

### Fold preparation
Create folds in a stratified fashion with the augmentations for train, val and test sets.

```bash 
python prepare_folds.py -dn dataset_name -dp /path/to/preprocessed/data -lp /path/to/label/data -cp /path/to/clinical/data -op /path/to/output/folder -dl desired_len
```
`dataset_name` should be 'NSCLC-Radiomics' for NSCLC-Radiomics dataset. It is not necessary for others. 

`/path/to/preprocessed/data` is the path to the preprocessed images.

`/path/to/label/data` is the path to the CSV file with at least two columns 'X' and 'y' for subject ids and labels.

`/path/to/clinical/data` is the path to the clinical data file (required for NSCLC-Radiomics).

`/path/to/output/folder` is the folder path to the output Excel file with several sheets with the following names : `Fold<fold_no>_<phase>`

`desired_len` is the desired number of instances for one class after the augmentations.

### Running experiments

```bash
python main.py -rp /paths/to/preprocesed/data -fp /paths/to/fold/files -resp result_file_name -exp exp_id -mode triplet/softmax -freeze
```

`/paths/to/preprocessed/data` should have the paths to the folders of preprocessed folders for all datasets. Paths should be seperated with a whitespace.

`/paths/to/fold/files` should have the paths to the Excel files with the folds. Paths should be seperated with a whitespace.
The file should have several sheets with the following names : `Fold<fold_no>_<phase>`

The results of the experiment will be saved to `./results/<result_file_name>.csv`

`mode` It should be either 'triplet' or 'softmax'. If it is 'triplet', the model will be trained with triplet loss. 
If it is 'softmax', the model will be trained with softmax loss.

`freeze` is an optional argument. If it is used, some layers of the model will be frozen.

Other parameters are set as default to the values used in the paper. 