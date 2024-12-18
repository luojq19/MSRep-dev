# MSRep

## Table of contents
- [MSRep](#msrep)
  - [Table of contents](#table-of-contents)
  - [Install dependencies](#install-dependencies)
  - [Download data](#download-data)
  - [Run MSRep to predict function annotatoins](#run-msrep-to-predict-function-annotatoins)
    - [Construct test and lookup database](#construct-test-and-lookup-database)
    - [Inference](#inference)
    - [Ensemble](#ensemble)
    - [Post-inference process for Gene Ontology](#post-inference-process-for-gene-ontology)
  - [Train MSRep](#train-msrep)
  - [Contact](#contact)

## Install dependencies
```
conda create -n msrep python=3.8
conda activate msrep
pip install -r requirements.txt
```

## Download data
Download the Swiss-Prot data for train/lookup from [Dropbox](https://www.dropbox.com/scl/fi/30izgj0zuyl8qnaw33tk9/data.zip?rlkey=0359ktdcsgdrk59gmqvwjjm2g&st=dyb88ij5&dl=0), unzip to `data` directory under the main directory.
```
wget https://www.dropbox.com/scl/fi/30izgj0zuyl8qnaw33tk9/data.zip?rlkey=0359ktdcsgdrk59gmqvwjjm2g&st=dyb88ij5&dl=0 -O data.zip
unzip data.zip
```

Download the pretrained MSRep model from [Dropbox](https://www.dropbox.com/scl/fi/8t91r3ggzfgdtlcmkltxj/checkpoints.zip?rlkey=pj79cjjqlzl08rrmixv75uswq&st=36bm1pwp&dl=0) and unzip it to `checkpoints` under the main directory.
```
wget https://www.dropbox.com/scl/fi/8t91r3ggzfgdtlcmkltxj/checkpoints.zip?rlkey=pj79cjjqlzl08rrmixv75uswq&st=36bm1pwp&dl=0 -O checkpoints.zip
unzip checkpoints.zip
```

Download precomputed ESM-1b embeddings from [Dropbox](https://www.dropbox.com/scl/fi/iexn4tpr243v7ydqagkuf/esm_embeddings.zip?rlkey=kvfm9k83gvk1p9xrxcm00gi1r&st=tfeb4jus&dl=0) and unzip it to `esm_embeddings` under the main directory.
```
wget https://www.dropbox.com/scl/fi/iexn4tpr243v7ydqagkuf/esm_embeddings.zip?rlkey=kvfm9k83gvk1p9xrxcm00gi1r&st=tfeb4jus&dl=0 -O esm_embeddings.zip
unzip esm_embeddings.zip
```

## Run MSRep to predict function annotatoins
### Construct test and lookup database
Organize the test proteins in a csv file with columns `Entry` and `Sequence` (See `data_cleaned/EC/swissprot_test_ec.csv` as an example). First construct test and lookup databases with ESM embeddings:
```
python scripts/extract_esm.py -i /path/to/test_csv -o /path/to/test_pt --ont ec
python scripts/extract_esm.py -i /path/to/lookup_csv -o /path/to/lookup_csv --ont ec
```
`-i`: input test csv file
`-o`: output test `pt` file
`--ont`: the function of interest, the available choices are `ec, gene3D, pfam, BP, MF, CC`.
You can use training data of MSRep as lookup data available in the `data` directory for each function anntation task (e.g., `data_cleaned/EC/swissprot_train_ec.csv`) or your own lookup data organized in a csv file with columns `Entry`, `Sequence`, and `Label`.

### Inference
Then running MSRep with the following command:
```
python scripts/predict.py configs/infer.yml
```
Parameters are set in the configuration file:
`model_dir`: path to the pretrained model directory, available in the `checkpoints` directory for each task. We provided five pretrained models with different random seed initialization for each task.
`lookup_data`: path to the lookup `pt` file
`test_data`: path to the test `pt` file.
`label_file`: list of all labels for the function, available as a json file in the `data` directory for each function.
`ont`: the function of interest, the available choices are `ec, gene3D, pfam, BP, MF, CC`.

### Ensemble
Run the majority ensemble for a list of prediction results with the following command:
```
python scripts/ensemble_majority_voting.py configs/ensemble.yml
```
Parameters are set in the configuratoin file:
`prediction_files`: list of paths to the prediction results generated by MSRep.

### Post-inference process for Gene Ontology
Following the convention of the CAFA challenge, we process the prediction results of Gene Ontology after the inference described above.

First download the hierarchy file of GO
```
wget http://purl.obolibrary.org/obo/go/go-basic.obo
```
Then conduct backpropagation on the original prediction result made by MSRep:
```
python scripts/flatten_GO.py -i /path/to/prediction -o /path/to/flat_prediction
python scripts/backprop.py --go_obo /path/to/go-basics.obo -i /path/to/flat_prediction
```
To ensemble the backpropagated predictions, run
```
python scripts/ensemble_GO.py configs/ensemble_GO.yml
```
Parameters are set in the configuratoin file:
`prediction_files`: list of paths to the backpropagated prediction results generated by MSRep.

## Train MSRep
You can train new MSRep models with your own data. First construct training .pt files following the commands in [database construction](#construct-test-and-lookup-database). And run the following command:
```
python scripts/train_MSRep.py /path/to/config
```
The example configuration files for training on different function annotation tasks are available in the `configs` directory of format `train_*.yml`, replace the dataset paths with the generated `pt` data files.

## Contact
Please submit GitHub issues or contact Jiaqi Luo (jiaqi@gatech.edu) and Yunan Luo (yunan@gatech.edu) for any questions related to the source code.
