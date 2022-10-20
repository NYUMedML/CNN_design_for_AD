# Generalizable deep learning model for early Alzheimer’s disease detection from structural MRIs
This repository contains code for a medical [paper](https://www.nature.com/articles/s41598-022-20674-x) and a machine learning [paper](http://proceedings.mlr.press/v116/liu20a) on deep learning for dementia.
In the medical [paper](https://www.nature.com/articles/s41598-022-20674-x), we compared the deep learning model with volume/thickness models on external independent cohort from [NACC](https://naccdata.org/). The volume and thickness data are extracted using the Freesurfer and quality controled by radiologists. 

If you would like to access the volume and thickness data as well as the subject and scan ID, please download it from the [/Data](https://github.com/NYUMedML/CNN_design_for_AD/tree/master/Data) folder.
<p float="left" align="center">
<img src="overview.png" width="800" /> 
<figcaption align="center">
Figure: Overview of the deep learning framework and performance for Alzheimer’s automatic diagnosis. (a) Deep learning framework used for automatic diagnosis. 
  

  **Contact:** [Sheng Liu](https://shengliu66.github.io/)


## Introduction
In this project, we focus on how to design CNN for Alzheimer's detection. we provide evidence that 
* instance normalization outperforms batch normalization  
* early spatial downsampling negatively affects performance
* widening the model brings consistent gains while increasing the depth does not
* incorporating age information yields moderate improvement. 
  
Compare with the volume/thickness model, the deep-learning model is
  * accurate
  * significantly faster than the volume/thickness model in which the volumes and thickness need to be extracted beforehand. 
  * can also be used to forecast progression:
  * relies on a wide range of regions associated with Alzheimer's disease. 
  * can automatically learn to identify imaging biomarkers that are predictive of Alzheimer's disease, and leverage them to achieve accurate early detection of the disease.


Together, these insights yield an increment of approximately 14% in test accuracy over existing models.
<!--   
<p float="left" align="center">
<img src="data_examples/visualization_02.png" width="200" /> 
<img src="data_examples/visualization_01.png" width="200" /> 
<img src="data_examples/visualization_03.png" width="200" /> 
</p> -->

<p float="left" align="center">
<img src="all_resized.gif" width="500" /> 
<figcaption align="center">
Figure 1. Visualization of the aggregated importance of each voxel (in yellow) in the deep learning model when classifying subjects into Cognitive Normal, Mild Cognitive Impairement, and Alzheimer's Disease. 


## Prerequisites
- Python 3.6
- PyTorch 0.4
- torchvision
- progress
- matplotlib
- numpy
- visdom

## License
This repository is licensed under the terms of the GNU AGPLv3 license.

## Download ADNI data
1. Request approval and register at [ADNI website](http://adni.loni.usc.edu/data-samples/access-data/)
2. Download both the scans and the clinical data. From the main page click on `PROJECTS` and `ADNI`. To download the imaging data, click on `Download` and choose `Image collections`. In the `Advanced search` tab, untick `ADNI 3` and tick `MRI` to download all the MR images.
3. In the `Advanced search results` tab, click Select `All` and `Add To Collection`. Finally, in the `Data Collection` tab, select the collection you just created, tick `All` and click on `Advanced download`. We advise you to group files as 10 zip files. To download the clinical data, click on `Download` and choose `Study Data`. Select all the csv files which are present in `ALL` by ticking Select `ALL `tabular data and click Download.

## Data Preprocessing
Data Preprocessing with Clinica:
1. **Convert data into BIDS format**: please read the docs on [Clinica website](http://www.clinica.run/doc/DatabasesToBIDS/#adni-to-bids), and install required softwares and download the required clinical files. Note that we first preprocess the training set to generate the template and use the template to preprocess validation and test set. You can find the [link](https://drive.google.com/file/d/1KurgyjQP-KReEO0gf31xxjwE5R-xuSRB/view?usp=sharing) to download the template we used for data preprocessing. You can find the script we use to run the converter at /datasets/files:
```
run_convert.sh
```

2. **preprocess converted and splitted data**: you can refer our scripts at /datasets/files. For training data, refer:
```
run_adni_preprocess.sh
```
For val and test refer:
```
run_adni_preprocess_val.sh
```
and 
```
run_adni_preprocess_test.sh
```

## Examples in the preprocessed dataset
Here are some examples of scans for each categories in our test dataset:

<p align="center">
<img src="data_examples/CN_example.png" width="600" /> 
<img src="data_examples/MCI_example.png" width="600" /> 
<img src="data_examples/AD_example.png" width="600" /> 
</p>

## Neural Network Training
Train the network ADNI dataset:

```
python main.py
```

You can create your own config files and add a **--config** flag to indicate the name of your config files.

## Model Evaluation
We provide the evaluation code in **Model_eval.ipynb**, where you can load and evaluate our trained model. The trained best model (with widening factor 8 and adding age) can be found [here](https://drive.google.com/file/d/1zU21Kin9kXg_qmj7w_u5dGOjXf1D5fa7/view?usp=sharing). 


## Results
<center>

| Dataset           | ADNI held-out        | ADNI held-out          | NACC external validation | NACC external validation |
| ----------------- | -------------------- | ---------------------- | -----------------------  | ------------------------ | 
|   Model           | Deep Learning model  | Volume/thickness model | Deep Learning model      | Volume/thickness model   |
| Cognitively Normal              | 87.59     | 84.45          | 85.12       | 80.77       |
| Mild Cognitive Impairment       | 62.59     | 56.95          | 62.45       | 57.88       |
| Alzheimer’s Disease Dementia    | 89.21     | 85.57          | 89.21       | 81.03       |
</center>
  
Table 1: Classifcation performance in ADNI held-out set and an external validation set. Area under ROC
curve for classifcation performance based on the  learning model vs the ROI-volume/thickness model,
for ADNI held-out set and NACC external validation set. Deep learning model outperforms ROI-volume/
thickness-based model in all classes. Please refer [paper](https://www.nature.com/articles/s41598-022-20674-x) for more details.

<p float="left" align="center">
<img src="AD_progression_new.png" width="800" /> 
<figcaption align="center">  
Figure: Progression analysis for MCI subjects. The subjects in the ADNI test set are divided
into two groups based on the classifcation results of the deep learning model from their frst scan diagnosed
as MCI: group A if the prediction is AD, and group B if it is not. The graph shows the fraction of subjects that
progressed to AD at diferent months following the frst scan diagnosed as MCI for both groups. Subjects in
group A progress to AD at a signifcantly faster rate, suggesting that the features extracted by the deep-learning
model may be predictive of the transition. 

<center>

| Method             | Acc.        | Balanced Acc. | Micro-AUC  | Macro-AUC |
| ----------------- | ----------- | ----------- | -----------  | ----------- | 
| ResNet-18 3D    | 52.4%      | 53.1%           | -           | -           |
| AlexNet 3D      | 57.2%      | 56.2%           | 75.1%       | 74.2%       |
| X 1             | 56.4%      | 54.8%           | 74.2%       | 75.6%       |
| X 2             | 58.4%      | 57.8%           | 77.2%       | 76.6%       |
| X 4             | 63.2%      | 63.3%           | 80.5%       | 77.0%       |
| X 8             | 66.9%      | 67.9%           | 82.0%       | 78.5%       |
| **X 8 + age**       | 68.2%      | 70.0%           | 82.0%       | 80.0%       |

</center>
  
Table 2: Classifcation performance in ADNI held-out with different neural network architectures. Please refer [paper](http://proceedings.mlr.press/v116/liu20a) for more details.
  
  
## References
  
```
@article{liu2022generalizable,
  title={Generalizable deep learning model for early Alzheimer’s disease detection from structural MRIs},
  author={Liu, Sheng and Masurkar, Arjun V and Rusinek, Henry and Chen, Jingyun and Zhang, Ben and Zhu, Weicheng and Fernandez-Granda, Carlos and Razavian, Narges},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```
  
```
@inproceedings{liu2020design,
  title={On the design of convolutional neural networks for automatic detection of Alzheimer’s disease},
  author={Liu, Sheng and Yadav, Chhavi and Fernandez-Granda, Carlos and Razavian, Narges},
  booktitle={Machine Learning for Health Workshop},
  pages={184--201},
  year={2020},
  organization={PMLR}
}
```
