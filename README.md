# SNP-Pro

SNP-Pro is a deep learning model designed for promoter activity prediction.   
- **Input data:** Personalized promoter sequence
- **Label data:** Promoter activity signal measured in SuRE library

    
## Requirement
- Python 3.6
- TensorFlow 1.9.0
- Cuda 10.0
   
   
## Model structure
SNP-Pro consists of two parellel feature embedding module and final prediction module.   
![Model Structure](https://user-images.githubusercontent.com/72458731/127801731-35a29e92-bb1a-4859-82d9-6bf3d189dd0b.jpg)


## Training
#### Step 1. Unzip feature files
```
cd ./data/features
tar -xvzf GM18983.tar.gz
```

#### Step 2. Run training script
```
python ./script/model/train.py -t [training set file] -a [validation set file] -d [feature file directory] -v [model version] 2> [stderr output file]
```
Example
```
python ./script/model/train.py -t ./dataset/GM18983/10000/cv1_tr.csv -a ./dataset/GM18983/10000/cv1_va.csv -d ./data/features/GM18983 -v d10000.cv1 2> d10000.cv1.log
```     
     
      
## Evaluation & Prediction
#### Step 1. Unzip feature files
```
cd ./data/features
tar -xvzf HG02601.tar.gz
```
    
#### Step 2. Run evaluation and predict script
```
python ./script/model/evaluation.py -i [test set file] -d [feature file directory ] -v [model version] --restore
python ./script/model/predict.py -i [test set file] -d [feature file directory ] -v [model version] --restore
```
Example
```
python ./script/model/evaluation.py -i ./dataset/HG02601/HG02601_20000.csv -d ./data/features/HG02601 -v d10000.cv1 --restore
python ./script/model/predict.py -i ./dataset/HG02601/HG02601_20000.csv -d ./data/features/HG02601 -v d10000.cv1 --restore

```
