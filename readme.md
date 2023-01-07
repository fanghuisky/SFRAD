
## Low-Shot Unsupervised Visual Anomaly Detection via Sparse Feature Representation (SFRAD)

This repository provides python implementation of 
the Sparse Feature Representation for Anomaly Detection (SFRAD) 
algorithms, which can achieve up to 99.1% image-level anomaly detection
ROC-AUC, 98.3% pixel-level anomaly segmentation ROC-AUC, and 93.6%
region-level PRO score.


## Dependencies
Our results were computed using Python 3.6, with packages and respective version noted in
`requirements.txt`. 

## Running the Experiments
### Full_shot
For full_shot anomaly detection, we have provided `train.sh`.
You can modify some parameters in `args` or `configs/custom.yml`
to test your own dataset. 
Note that, the parameters in `configs.custom.yml` take precedence.
```shell
env PYTHONPATH=src python run_patchcore.py --config custom.yml
```

### Low-shot
For low-shot anomaly detection, we have provided some random number
in `low_shot_file`. 
The random number 
can be generated through `./low_shot_file/get_random.py`.
You can run the following script to test low-shot anomaly detection.
```shell
for k in $(seq 0 109)
do
    env PYTHONPATH=src python run_patchcore.py --config mvtec_low_shot.yml --label_times $k
done
```


## Dataset
If you use your dataset, make sure that it follows 
the following data tree:
```shell
dataset
|-- category1
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ anomaly1
|-----|--------|------ anomaly2
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
|-- category2
|-- ...
```

### Data download
MVTec AD dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad

KSDD: http://www.vicos.si/Downloads/KolektorSDD

KSDD2: https://paperswithcode.com/dataset/kolektorsdd2

STC dataset: https://svip-lab.github.io/dataset/campus_dataset.html



## This implementation is based on / inspired by :

https://github.com/amazon-science/patchcore-inspection (PatchCore)

https://github.com/ChongYou/subspace-clustering (SSC-OMP)