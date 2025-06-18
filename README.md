# Semi-Supervised U-Estimator 项目

## Conda Enviroment

```bash
conda env create -f environment.yml
conda activate semi-u
```
## Run Experiment
```bash
python main.py --experiment pairwise --n_labeled 200 --n_unlabeled 1000 --p 5 --discrete_levels 5 --trials 50
```

