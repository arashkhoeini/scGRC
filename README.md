# Single Cell Gene Regularized Clustering (scGRC)


## Requirements
* Python 3.6 
* PyTorch 1.9.1
* scanpy 1.7.2
* anndata 0.7.6

Setting up the virtual environment:
```
pytrhon3.6 -m venv ENV
source ENV/bin/activate
pip install -r requirements.txt
```

## Datasets 
1.`Synthetic`  [[datasets/simulated](https://github.com/arashkhoeini/scGRC/tree/main/datasets/simulated)]
2.`Tabula Muris` [[download link](https://figshare.com/projects/Tabula_Muris_Transcriptomic_characterization_of_20_organs_and_tissues_from_Mus_musculus_at_single_cell_resolution/27733)]


## Running Experiments

**Note**: make sure that you first set the dataset path in the corresponding YAML file (for Tabula Muris download it using the download link provided above, in the **Datasets** section)

### Explainability results using Simulated
```
python ./main_simulated.py 
```

### ARI/ACC using Tabula Muris

```
python ./main_tabulamuris.py 
```

