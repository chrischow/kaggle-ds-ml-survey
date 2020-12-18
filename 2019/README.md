
# EDA for 2019 Survey Data


```python
# Import required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

# Settings
sns.set()
warnings.filterwarnings('ignore')

# Global params
title_fd = {'fontsize': 15, 'fontweight': 'bold'}
```

## Import Data


```python
# Load data
df = pd.read_csv('multiple_choice_responses.csv')

# Extract definitions
qns = df.loc[0, :].copy().T
df = df.loc[1:, :]
```

## Helper Functions


```python
# Plot bar
def plot_bar(col, title, sortorder=None, horizontal=True, figsize=(6.4,4.8)):
    if sortorder:
        data = df[col].value_counts(normalize=True).loc[sortorder]
    else:
        data = df[col].value_counts(normalize=True).sort_values(ascending=False)
    if horizontal:
        data.plot.barh(figsize=figsize)
        plt.gca().invert_yaxis()
    else:
        data.plot.bar(figsize=figsize)
    plt.title(title, fontdict=title_fd)
    plt.show()

# Plot multiple select
def plot_multi(qn, title, names, nonecol, dropnone=True, horizontal=True, figsize=(6.4, 4.8)):
    cols = qns.index[qns.index.str.contains(qn)][:-1]
    
    # Breakdown
    temp_data = df[cols].copy()
    temp_data.loc[temp_data.notnull().sum(axis=1) == 0, f'{qn}_Part_{nonecol}'] = 'None'
    if dropnone:
        temp_data = temp_data.drop(f'{qn}_Part_{nonecol}', axis=1)
        final_names = names[:-2] + [names[-1]]
    else:
        final_names = names
    data = temp_data.notnull().sum()
    data.index = final_names
    data.sort_values(ascending=False).plot.barh(figsize=figsize)
    plt.gca().invert_yaxis()
    plt.title(title, fontdict=title_fd)
    plt.show()

    # Number of selections
    temp_data = df[cols].copy()
    if nonecol:
        temp_data = temp_data.drop(f'{qn}_Part_{nonecol}', axis=1)
    data = pd.Series(temp_data.notnull().sum(axis=1)).value_counts().sort_index()
    # data = data.drop(0)
    data.plot.bar()
    plt.title(f'No. of {title}', fontdict=title_fd)
    plt.show()
```

## Data Exploration


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time from Start to Finish (seconds)</th>
      <th>Q1</th>
      <th>Q2</th>
      <th>Q2_OTHER_TEXT</th>
      <th>Q3</th>
      <th>Q4</th>
      <th>Q5</th>
      <th>Q5_OTHER_TEXT</th>
      <th>Q6</th>
      <th>Q7</th>
      <th>...</th>
      <th>Q34_Part_4</th>
      <th>Q34_Part_5</th>
      <th>Q34_Part_6</th>
      <th>Q34_Part_7</th>
      <th>Q34_Part_8</th>
      <th>Q34_Part_9</th>
      <th>Q34_Part_10</th>
      <th>Q34_Part_11</th>
      <th>Q34_Part_12</th>
      <th>Q34_OTHER_TEXT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>510</td>
      <td>22-24</td>
      <td>Male</td>
      <td>-1</td>
      <td>France</td>
      <td>Master’s degree</td>
      <td>Software Engineer</td>
      <td>-1</td>
      <td>1000-9,999 employees</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>423</td>
      <td>40-44</td>
      <td>Male</td>
      <td>-1</td>
      <td>India</td>
      <td>Professional degree</td>
      <td>Software Engineer</td>
      <td>-1</td>
      <td>&gt; 10,000 employees</td>
      <td>20+</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>83</td>
      <td>55-59</td>
      <td>Female</td>
      <td>-1</td>
      <td>Germany</td>
      <td>Professional degree</td>
      <td>NaN</td>
      <td>-1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>391</td>
      <td>40-44</td>
      <td>Male</td>
      <td>-1</td>
      <td>Australia</td>
      <td>Master’s degree</td>
      <td>Other</td>
      <td>0</td>
      <td>&gt; 10,000 employees</td>
      <td>20+</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Azure SQL Database</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>392</td>
      <td>22-24</td>
      <td>Male</td>
      <td>-1</td>
      <td>India</td>
      <td>Bachelor’s degree</td>
      <td>Other</td>
      <td>1</td>
      <td>0-49 employees</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 246 columns</p>
</div>



## Personal Info


```python
sortorder = [
    '18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
    '55-59', '60-69', '70+'
]

plot_bar('Q1', 'Age Profile', sortorder, False)
plot_bar('Q2', 'Gender Profile')
plot_bar('Q4', 'Education Profile')
plot_bar('Q5', 'Job Title')

sortorder = [
    'I have never written code', '< 1 years', '1-2 years', '3-5 years',
    '5-10 years', '10-20 years', '20+ years'
]

plot_bar('Q15', 'Coding Experience', sortorder)

sortorder = [
    '< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years',
    '5-10 years', '10-15 years', '20+ years'
]

plot_bar('Q23', 'Experience in ML Methods', sortorder)
```


![png](/2019/images/output_9_0.png)



![png](/2019/images/output_9_1.png)



![png](/2019/images/output_9_2.png)



![png](/2019/images/output_9_3.png)



![png](/2019/images/output_9_4.png)



![png](/2019/images/output_9_5.png)



```python

```


![png](/2019/images/output_10_0.png)


## Company Details


```python
sortorder = [
    '0-49 employees', '50-249 employees', '250-999 employees',
    '1000-9,999 employees', '> 10,000 employees', 
    
]

plot_bar('Q6', 'Company Size', sortorder=sortorder)

sortorder = [
    '0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+'
]

plot_bar('Q7', 'No. of Individuals Responsible for DS Workloads', sortorder=sortorder)

sortorder = [
    'We are exploring ML methods (and may one day put a model into production)',
    'We recently started using ML methods (i.e., models in production for less than 2 years)',
    'We have well established ML methods (i.e., models in production for more than 2 years)',
    'No (we do not use ML methods)',
    'We use ML methods for generating insights (but do not put working models into production)',
    'I do not know'
]

plot_bar('Q8', 'Incorporation of ML Methods', sortorder=sortorder)

names = [
    'Analyze and understand data to influence product or business decisions',
    'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
    'Build prototypes to explore applying machine learning to new areas',
    'Build and/or run a machine learning service that operationally improves my product or workflows',
    'Experimentation and iteration to improve existing ML models',
    'Do research that advances the state of the art of machine learning',
    'None of these activities are an important part of my role at work',
    'Other'
]

plot_multi('Q9', 'Activities at Work', names, len(names)-1)

sortorder = [
    '$0-999', '10,000-14,999', '100,000-124,999', '30,000-39,999',
    '40,000-49,999', '50,000-59,999', '1,000-1,999', '60,000-69,999',
    '5,000-7,499', '15,000-19,999', '20,000-24,999', '70,000-79,999',
    '125,000-149,999', '25,000-29,999', '150,000-199,999', '7,500-9,999',
    '80,000-89,999', '2,000-2,999', '90,000-99,999', '3,000-3,999',
    '4,000-4,999', '200,000-249,999', '> $500,000', '300,000-500,000',
    '250,000-299,999'
]

plot_bar('Q10', 'Yearly Compensation (USD)', sortorder, figsize=(6.4,8))

sortorder = [
    '$0 (USD)', '$1-$99', '$100-$999', '$1000-$9,999', '$10,000-$99,999',
   '> $100,000 ($USD)'
]
plot_bar('Q11', 'Expenditure on ML / Cloud Computing (USD) in Past 5 years', )
```


![png](/2019/images/output_12_0.png)



![png](/2019/images/output_12_1.png)



![png](/2019/images/output_12_2.png)



![png](/2019/images/output_12_3.png)



![png](/2019/images/output_12_4.png)



![png](/2019/images/output_12_5.png)



![png](/2019/images/output_12_6.png)


## Programming Languages


```python
names = [
    'Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript',
    'TypeScript', 'Bash', 'MATLAB', 'None', 'Other'
]

plot_multi('Q18', 'Programming Languages', names, len(names)-1)
plot_bar('Q19', 'Recommended Programming Language to Learn First')
```


![png](/2019/images/output_14_0.png)



![png](/2019/images/output_14_1.png)



![png](/2019/images/output_14_2.png)


## Programming IDEs


```python
names = [
    'Jupyter (JupyterLab, Jupyter Notebooks, etc)', 'RStudio',
    'PyCharm', 'Atom', 'MATLAB',
    'Visual Studio / Visual Studio Code', 'Spyder',
    'Vim / Emacs', 'Notepad++', 'Sublime Text', 'None',
    'Other'
]

plot_multi('Q16', 'Programming IDEs', names, len(names)-1)

names = [
    'Kaggle Notebooks (Kernels)', 'Google Colab',
    'Microsoft Azure Notebooks',
    'Google Cloud Notebook Products (AI Platform, Datalab, etc)',
    'Paperspace / Gradient', ' FloydHub ', ' Binder / JupyterHub',
    'IBM Watson Studio', 'Code Ocean',
    'AWS Notebook Products (EMR Notebooks, Sagemaker Notebooks, etc)',
    'None', 'Other'
]

plot_multi('Q17', 'Hosted Notebooks', names, len(names)-1)
```


![png](/2019/images/output_16_0.png)



![png](/2019/images/output_16_1.png)



![png](/2019/images/output_16_2.png)



![png](/2019/images/output_16_3.png)


## Data Visualisation Libraries


```python
names = [
    'Ggplot / ggplot2 ', 'Matplotlib ', 'Altair ', 'Shiny ', 'D3.js ',
    'Plotly / Plotly Express ', 'Bokeh ', 'Seaborn ', 'Geoplotlib ',
    'Leaflet / Folium ', 'None', 'Other'
]

plot_multi('Q20', 'Data Visualisation Libraries', names, len(names)-1)
```


![png](/2019/images/output_18_0.png)



![png](/2019/images/output_18_1.png)


## ML Frameworks


```python
names = [
    'Scikit-learn', 'TensorFlow', 'Keras', 'RandomForest',
    'Xgboost', 'PyTorch', 'Caret', 'LightGBM', 'Spark MLib',
    'Fast.ai',
    'None',
    'Other'
]

plot_multi('Q28', 'ML Frameworks', names, len(names)-1)
```


![png](/2019/images/output_20_0.png)



![png](/2019/images/output_20_1.png)


## Big Data Products


```python
names = [
    'Google BigQuery', 'AWS Redshift', 'Databricks',
    'AWS Elastic MapReduce', 'Teradata', 'Microsoft Analysis Services',
    'Google Cloud Dataflow', 'AWS Athena', 'AWS Kinesis',
    'Google Cloud Pub/Sub', 'None', 'Other'
]

plot_multi('Q31', 'Big Data Products', names, len(names)-1)

names = [
    'MySQL', 'PostgresSQL', 'SQLite', 'Microsoft SQL Server',
    'Oracle Database', 'Microsoft Access',
    'AWS Relational Database Service', 'AWS DynamoDB',
    'Azure SQL Database', 'Google Cloud SQL', 'None', 'Other'
]

plot_multi('Q34', 'Relational Database Products', names, len(names)-1)
```


![png](/2019/images/output_22_0.png)



![png](/2019/images/output_22_1.png)



![png](/2019/images/output_22_2.png)



![png](/2019/images/output_22_3.png)


## Automated ML Tools


```python
names = [
    'Automated data augmentation (e.g. imgaug, albumentations)',
    'Automated feature engineering/selection (e.g. tpot, boruta_py)',
    'Automated model selection (e.g. auto-sklearn, xcessiv)',
    'Automated model architecture searches (e.g. darts, enas)',
    'Automated hyperparameter tuning (e.g. hyperopt, ray.tune)',
    'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)',
    'None', 'Other'
]

plot_multi('Q25', 'Automated ML Processes', names, len(names)-1)

names = [
    'Google AutoML', 'H20 Driverless AI', 'Databricks AutoML',
    'DataRobot AutoML', 'Tpot', 'Auto-Keras', 'Auto-Sklearn',
    'Auto_ml', 'Xcessiv', 'MLbox',
    'None', 'Other'
]

plot_multi('Q33', 'Automated ML Tools', names, len(names)-1)
```


![png](/2019/images/output_24_0.png)



![png](/2019/images/output_24_1.png)



![png](/2019/images/output_24_2.png)



![png](/2019/images/output_24_3.png)



```python

```


```python

```


```python
names = [
    'Udacity', 'Coursera', 'edX', 'DataCamp', 'DataQuest',
    'Kaggle Courses (i.e. Kaggle Learn)', 'Fast.ai', 'Udemy',
    'LinkedIn Learning',
    'University Courses (resulting in a university degree)', 'None',
    'Other'
]
plot_multi('Q13', 'DS Learning Platforms', names, len(names)-1)
```


![png](/2019/images/output_27_0.png)



![png](/2019/images/output_27_1.png)



```python
plot_bar('Q14', 'Primary Tools Used to Analyse Data')
```


![png](/2019/images/output_28_0.png)



```python
names = [
    'CPUs', 'GPUs', 'TPUs', 'None / I do not know', 'Other'
]

plot_multi('Q21', 'Hardware Types', names, len(names)-1)
```


![png](/2019/images/output_29_0.png)



![png](/2019/images/output_29_1.png)



```python
names = [
    'Linear or Logistic Regression',
    'Decision Trees or Random Forests',
    'Gradient Boosting Machines (xgboost, lightgbm, etc)',
    'Bayesian Approaches', 'Evolutionary Approaches',
    'Dense Neural Networks (MLPs, etc)',
    'Convolutional Neural Networks', 'Generative Adversarial Networks',
    'Recurrent Neural Networks',
    'Transformer Networks (BERT, gpt-2, etc)', 'None', 'Other'
]

plot_multi('Q24', 'ML Algorithms', names, len(names)-1)
```


![png](/2019/images/output_30_0.png)



![png](/2019/images/output_30_1.png)



```python
names = [
    'General purpose image/video tools (PIL, cv2, skimage, etc)',
    'Image segmentation methods (U-Net, Mask R-CNN, etc)',
    'Object detection methods (YOLOv3, RetinaNet, etc)',
    'Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)',
    'Generative Networks (GAN, VAE, etc)', 'None', 'Other'
]

plot_multi('Q26', 'Computer Vision Methods', names, len(names)-1)
```


![png](/2019/images/output_31_0.png)



![png](/2019/images/output_31_1.png)



```python
names = [
    'Word embeddings/vectors (GLoVe, fastText, word2vec)',
    'Encoder-decorder models (seq2seq, vanilla transformers)',
    'Contextualized embeddings (ELMo, CoVe)',
    'Transformer language models (GPT-2, BERT, XLnet, etc)', 'None',
    'Other'
]

plot_multi('Q27', 'NLP Methods', names, len(names)-1)
```


![png](/2019/images/output_32_0.png)



![png](/2019/images/output_32_1.png)



```python
names = [
    'Google Cloud Platform (GCP)', 'Amazon Web Services (AWS)',
    'Microsoft Azure', 'IBM Cloud', 'Alibaba Cloud',
    'Salesforce Cloud', 'Oracle Cloud', 'SAP Cloud',
    'VMware Cloud', 'Red Hat Cloud',
    'None',
    'Other'
]

plot_multi('Q29', 'Cloud Computing Platforms', names, len(names)-1)
```


![png](/2019/images/output_33_0.png)



![png](/2019/images/output_33_1.png)



```python
names = [
    'AWS Elastic Compute Cloud (EC2)', 'Google Compute Engine (GCE)',
    'AWS Lambda', 'Azure Virtual Machines', 'Google App Engine',
    'Google Cloud Functions', 'AWS Elastic Beanstalk',
    'Google Kubernetes Engine', 'AWS Batch', 'Azure Container Service',
    'None', 'Other'
]

plot_multi('Q30', 'Cloud Computing Products', names, len(names)-1)
```


![png](/2019/images/output_34_0.png)



![png](/2019/images/output_34_1.png)



```python
names = [
    'SAS', 'Cloudera', 'Azure Machine Learning Studio',
    'Google Cloud Machine Learning Engine', 'Google Cloud Vision',
    'Google Cloud Speech-to-Text', 'Google Cloud Natural Language',
    'RapidMiner', 'Google Cloud Translation', 'Amazon SageMaker',
    'None', 'Other'
]

plot_multi('Q32', 'ML Products', names, len(names)-1)
```


![png](/2019/images/output_35_0.png)



![png](/2019/images/output_35_1.png)



```python
names = [
    'Twitter (data science influencers)',
    'Hacker News (https://news.ycombinator.com/)',
    'Reddit (r/machinelearning, r/datascience, etc)',
    'Kaggle (forums, blog, social media, etc)',
    'Course Forums (forums.fast.ai, etc)',
    'YouTube (Cloud AI Adventures, Siraj Raval, etc)',
    'Podcasts (Chai Time Data Science, Linear Digressions, etc)',
    'Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)',
    'Journal Publications (traditional publications, preprint journals, etc)',
    'Slack Communities (ods.ai, kagglenoobs, etc)', 'None', 'Other'
]
plot_multi('Q12', 'Favourite Media Source for DS Topics', names, len(names)-1)
```


![png](/2019/images/output_36_0.png)



![png](/2019/images/output_36_1.png)



```python
# Helper code
column = 'Q34'

try:
    print(df[column].value_counts())
    print(df[column].value_counts().index)
except:
    pass
qns[qns.index.str.contains(column)].str.replace(
    "Which of the following relational database products do you use on a regular basis\\? \\(Select all that apply\\) - Selected Choice - ",
    '').values
```




    array(['MySQL', 'PostgresSQL', 'SQLite', 'Microsoft SQL Server',
           'Oracle Database', 'Microsoft Access',
           'AWS Relational Database Service', 'AWS DynamoDB',
           'Azure SQL Database', 'Google Cloud SQL', 'None', 'Other',
           'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Other - Text'],
          dtype=object)




```python

```
