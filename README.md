
# Data Science (DS) Tools for Large Organisations
As small-to-medium-sized companies scale up, or as big companies (like mine) jump on the data bandwagon, some standardisation in tools is required to make sure that data scientists are speaking and coding in the same or similar languages. What better way to get inspiration for possible enterprise tools than from data science professionals and future data scientists? This notebook aims to use data from the [2020 Kaggle Data Science (DS) and Machine Learning (ML) survey](https://www.kaggle.com/c/kaggle-survey-2020/overview) to identify ideal tools for large enterprises.

## TL;DR
We looked at tools preferred by (1) employees in companies with 1,000 or more staff and that had 20 or more people managing data science workloads, and (2) students. This was to capture (i) tools currently being employed by large enterprises and (ii) tools that new employees would be familiar with upon entering the workforce. Based on the data, the top picks were:

| Type of Tool | Top Picks |
| :----------- | :-------- |
| Programming Languages | Python, SQL, and R |
| IDEs | Jupyter, Visual Studio Code, PyCharm, and RStudio |
| Business Intelligence | Tableau, Power BI, and Google Data Studio |
| Data Visualisation Libraries | Matplotlib, Seaborn, ggplot2, Plotly |
| ML Frameworks | As many as possible; Default: Scikit-Learn for Python, Caret for R |
| Big Data Products (Cloud Only) | Google Cloud BigQuery, Azure Data Lake Storage, and AWS Redshift |
| Databases | MySQL, Microsoft SQL Server, PostgreSQL, and MongoDB |
| Automated ML | Auto-SKlearn, Auto-Keras, and AutoML |
| ML Management (Open Source) | TensorBoard, Trains, and Sacred + Omniboard |
| ML Management (Paid; Cloud) | Neptune.ai, Weights & Biases, and Comet.ml |
| Sharing & Deployment | Shiny, Streamlit, and Dash |

## Approach
The types of tools that we aim to identify are:

1. Programming Languages
2. Integrated Development Environments (IDEs)
3. Business Intelligence
3. Data Visualisation (Dataviz) Libraries
4. ML Frameworks
5. Big Data Products
6. Automated ML
7. ML Management
8. Sharing and Deployment

To do so, we look at the responses of two groups of survey participants:

1. **Employees in companies with 1,000 or more staff *and* more than 20 people managing data science workloads.** This group captures the tools employed by large enterprises at the time of the survey.
2. **Students.** This group captures the tools that new employees would be familiar with when they enter the workforce.

<details><summary style="color: #8497B0;"><em>Setup Code</em></summary>
<p>

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
colors = ['#5E72EB', '#27DDCB']

# Helper function: Plot multiple
def plot_multi(qn, title, names, nonecol, dropnone=True, figsize=(8,6)):
    cols = qns.index[qns.index.str.contains(qn)]
    
    # Breakdown
    temp_data = df[cols].copy()
    temp_data.loc[temp_data.notnull().sum(axis=1) == 0, f'{qn}_Part_{nonecol}'] = 'None'
    
    if dropnone:
        temp_data = temp_data.drop(f'{qn}_Part_{nonecol}', axis=1)
        final_names = names[:-2] + [names[-1]]
    else:
        final_names = names
    
    data_le = temp_data[df.group == 'Large Enterprise'].notnull().sum()
    data_st = temp_data[df.group == 'Student'].notnull().sum()
    
    data_le = data_le / data_le.sum()
    data_st = data_st / data_st.sum()
    
    data_le.index = final_names
    data_st.index = final_names
    
    data = pd.concat([data_le, data_st], axis=1)
    data.columns = ['Large Enterprise', 'Student']
    
    data.sort_values('Large Enterprise', ascending=False).plot.barh(color=colors, figsize=figsize)
    plt.gca().invert_yaxis()
    plt.title(title, fontdict=title_fd)
    plt.show()
    
# Helper function: Plot pairs
def plot_pairs(tool, colnames, newnames):
    heatmap = []
    tool1 = []
    tool2 = []
    pair_count = []

    # Large enterprise employees
    df_le = df.loc[df.group == 'Large Enterprise', colnames]
    df_le.columns = newnames
    for i in newnames:
        for j in newnames:
            heatmap.append(
                np.sum(( df_le[i].notnull() ) & ( df_le[j].notnull() )) / np.sum( df_le[i].notnull() ) * 100
            )

            tool1.append(i)
            tool2.append(j)
            pair_count.append(np.sum(( df_le[i].notnull() ) & ( df_le[j].notnull() )))

    heatmap = np.array(heatmap).reshape(len(newnames), len(newnames))
    heatmap = pd.DataFrame(heatmap, columns=newnames, index=newnames)
    plt.figure(figsize=(12,10))
    sns.heatmap(heatmap, annot=True, cmap='RdBu_r')
    plt.title(f'{tool} Pairs - Large Enterprise Employees', fontdict=title_fd)
    plt.ylabel(f'First {tool}')
    plt.xlabel(f'Second {tool}')
    plt.show()

    df_pairs = pd.DataFrame({
        'tool1': tool1,
        'tool2': tool2,
        'pair_count': pair_count
    })
    df_pairs = df_pairs.loc[df_pairs.tool1 != df_pairs.tool2].sort_values('pair_count', ascending=False)
    df_pairs = df_pairs.drop_duplicates(subset=['pair_count']).reset_index(drop=True)
    display(df_pairs.head(10))

    # Students
    heatmap = []
    tool1 = []
    tool2 = []
    pair_count = []
    df_st = df.loc[df.group == 'Student', colnames]
    df_st.columns = newnames
    for i in newnames:
        for j in newnames:
            try:
                heatmap.append(
                    np.sum(( df_st[i].notnull() ) & ( df_st[j].notnull() )) / np.sum(df_st[i].notnull()) * 100
                )
            except:
                heatmap.append(0)

            tool1.append(i)
            tool2.append(j)
            pair_count.append(np.sum(( df_st[i].notnull() ) & ( df_st[j].notnull() )))

    heatmap = np.array(heatmap).reshape(len(newnames), len(newnames))
    heatmap = pd.DataFrame(heatmap, columns=newnames, index=newnames)
    plt.figure(figsize=(12,10))
    sns.heatmap(heatmap, annot=True, cmap='RdBu_r')
    plt.title(f'{tool} Pairs - Student', fontdict=title_fd)
    plt.ylabel(f'First {tool}')
    plt.xlabel(f'Second {tool}')
    plt.show()

    df_pairs = pd.DataFrame({
        'tool1': tool1,
        'tool2': tool2,
        'pair_count': pair_count
    })
    df_pairs = df_pairs.loc[df_pairs.tool1 != df_pairs.tool2].sort_values('pair_count', ascending=False)
    df_pairs = df_pairs.drop_duplicates(subset=['pair_count']).reset_index(drop=True)
    display(df_pairs.head(10))
```

</p>
</details>

## Data Preparation
The data provided was already pretty clean. All I had to do was add a group to indicate the groups we were interested in, and delete all other entries.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
# Load data
df = pd.read_csv('2020/kaggle_survey_2020_responses.csv')

# Extract definitions
qns = df.loc[0, :].copy().T
df = df.loc[1:, :]

# Create groups
df['group'] = ''
df.loc[df.Q5 == 'Student', 'group'] = 'Student'
df.loc[(df.Q20.isin(['1000-9,999 employees', '10,000 or more employees'])) & (df.Q21=='20+'), 'group'] = 'Large Enterprise'

# Remove respondents not belonging to any of these groups
df = df[df.group != '']
```

</p>
</details>

## Comparison of Groups
To better understand each group, we compared the two groups in terms of age, gender, coding experience, and ML experience. We see that:

* There were more student respondents than large enterprise employees (5,171 vs. 1,877)
* The large enterprise employees were generally older and more experienced
* The proportions of men and women from both groups were similar

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
df.group.value_counts().plot.barh(color=colors[::-1])
plt.title('No. of Respondents', fontdict=title_fd)
plt.show()

grp_age = pd.concat([df.loc[df.group == 'Large Enterprise', 'Q1'].value_counts(normalize=True), df.loc[df.group == 'Student', 'Q1'].value_counts(normalize=True)], axis=1).sort_index()
grp_age.columns=['Large Enterprise', 'Student']
grp_age.plot.bar(color=colors)
plt.title('Age by Group', fontdict=title_fd)
plt.show()

grp_gender = pd.concat([df.loc[df.group == 'Large Enterprise', 'Q2'].value_counts(normalize=True), df.loc[df.group == 'Student', 'Q2'].value_counts(normalize=True)], axis=1).sort_index()
grp_gender.columns=['Large Enterprise', 'Student']
grp_gender.plot.barh(color=colors)
plt.title('Gender by Group', fontdict=title_fd)
plt.gca().invert_yaxis()
plt.show()

grp_coding_exp = pd.concat([df.loc[df.group == 'Large Enterprise', 'Q6'].value_counts(normalize=True), df.loc[df.group == 'Student', 'Q6'].value_counts(normalize=True)], axis=1)
grp_coding_exp.columns=['Large Enterprise', 'Student']
grp_coding_exp = grp_coding_exp.loc[['I have never written code', '< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']]
grp_coding_exp.plot.barh(color=colors)
plt.title('Coding Experience by Group', fontdict=title_fd)
plt.gca().invert_yaxis()
plt.show()

grp_ml_exp = pd.concat([df.loc[df.group == 'Large Enterprise', 'Q15'].value_counts(normalize=True), df.loc[df.group == 'Student', 'Q15'].value_counts(normalize=True)], axis=1)
grp_ml_exp.columns=['Large Enterprise', 'Student']
grp_ml_exp = grp_ml_exp.loc[['I do not use machine learning methods', 'Under 1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-20 years', '20 or more years']]
grp_ml_exp.plot.barh(color=colors)
plt.title('ML Experience by Group', fontdict=title_fd)
plt.gca().invert_yaxis()
plt.show()
```

</p></details>

![No. of Respondents by Group](/images/output_7_0.png)



![Age by Group](/images/output_7_1.png)



![Gender by Group](/images/output_7_2.png)



![Coding Experience by Group](/images/output_7_3.png)



![ML Experience by Group](/images/output_7_4.png)


## Programming Languages
The rank for programming languages differed among the two groups:

| Rank | Large Enterprise | Student |
| :--: | :--------------: | :-----: |
| 1    | Python           | Python  |
| 2    | SQL              | **C++** |
| 3    | R                | **C**   |
| 4    | Java             | SQL     |
| 5    | Bash             | Java    |

It's possible that we observed this result because of selection bias. Student respondents who saw Kaggle's ads for the survey (via mailing list, website, and Twitter) were more likely to be Computer Science students who actually have a Kaggle account, than students in other programmes like Engineering, Economics or the Arts. And of course, Computer Science students focus on lower-level languages like C/C++ in addition to Python. As more undergraduate DS programmes are stood up, we may see this order change.

If the above is true, we ought to take reference from the Large Enterprise group: **(1) Python, (2) SQL, and (3) R are the programming languages to focus on**.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB',
    'None', 'Other'
]

plot_multi('Q7', 'Programming Languages', names, len(names)-1)
```

</p></details>

![Programming Languages](/images/output_9_0.png)


Out of curiosity, I wanted to know which pair of programming languages was most popular. The two graphs below show the most popular pairs of languages for large enterprise employees and students, respectively. The "heatmaps" represent proportions of individuals who are familiar in a language on the horizontal axis, given that they are already familiar with the language given in the vertical axis. For example, in the 2nd square from the left in the top row (28%) of the large enterprise employees heatmaps: of all individuals who coded in Python, 28% of them also coded in R.

Here are some interesting findings:
* Regardless of "first" language, most people also knew Python (see 1st column of both graphs)
* It appears that more people switch from R to Python than Python to R:
    * Only 28% of those who knew Python also knew R
    * Meanwhile, 81% of those who knew R also knew Python

The top 3 pairs by absolute numbers for large enterprise employees were: (1) Python and SQL, (2) Python and R, and (3) R and SQL. Meanwhile, the top 3 pairs for students were (1) C++ and Python, (2) C and Python, and (3) Python and SQL. This doesn't change the conclusion from above that Python, R, and SQL are the most popular languages in large enterprises.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
# Create heatmap
heatmap = []
colnames = [
    'Q7_Part_1',
    'Q7_Part_2',
    'Q7_Part_3',
    'Q7_Part_4',
    'Q7_Part_5',
    'Q7_Part_6',
    'Q7_Part_7',
    'Q7_Part_8',
    'Q7_Part_9',
    'Q7_Part_10',
    'Q7_Part_11',
]

newnames = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB']

TOOL = 'Programming Language'

plot_pairs(TOOL, colnames, newnames)
```

</p></details>

![Programming Languages Pairs - Large Enterprise Employees](/images/output_11_0.png)



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
      <th>tool1</th>
      <th>tool2</th>
      <th>pair_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SQL</td>
      <td>Python</td>
      <td>880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Python</td>
      <td>R</td>
      <td>428</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SQL</td>
      <td>R</td>
      <td>332</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bash</td>
      <td>Python</td>
      <td>302</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Java</td>
      <td>Python</td>
      <td>285</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Python</td>
      <td>C++</td>
      <td>238</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Python</td>
      <td>Javascript</td>
      <td>236</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SQL</td>
      <td>Bash</td>
      <td>214</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SQL</td>
      <td>Java</td>
      <td>208</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C</td>
      <td>Python</td>
      <td>191</td>
    </tr>
  </tbody>
</table>
</div>



![Programming Languages Pairs - Students](/images/output_11_2.png)



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
      <th>tool1</th>
      <th>tool2</th>
      <th>pair_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C++</td>
      <td>Python</td>
      <td>1493</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>Python</td>
      <td>1352</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SQL</td>
      <td>Python</td>
      <td>1310</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Python</td>
      <td>Java</td>
      <td>1027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C++</td>
      <td>C</td>
      <td>855</td>
    </tr>
    <tr>
      <th>5</th>
      <td>R</td>
      <td>Python</td>
      <td>786</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Python</td>
      <td>MATLAB</td>
      <td>695</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Javascript</td>
      <td>Python</td>
      <td>658</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C</td>
      <td>Java</td>
      <td>583</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SQL</td>
      <td>C++</td>
      <td>568</td>
    </tr>
  </tbody>
</table>
</div>


## Integrated Development Environments (IDEs)
Respondents from both groups were in agreement on 4 of the top 5 programming IDEs: (1) Jupyter, (2) Visual Studio Code, (3) PyCharm, and (4/5) RStudio. For the 4th/5th IDE, the students preferred Spyder, while the large enterprise employees preferred Notepad++ for some reason (I dislike both of these choices).

| Rank | Large Enterprise | Student |
| :--: | :--------------: | :-----: |
| 1    | Jupyter          | Jupyter |
| 2    | VSCode           | VSCode  |
| 3    | PyCharm          | PyCharm |
| 4    | RStudio          | Spyder  |
| 5    | Notepad++        | RStudio |

The consensus top 3: **(1) Jupyter, (2) Visual Studio Code, and (3) PyCharm**. To cater for R users, I'd throw in **RStudio** as a fourth tool.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'Jupyter', 'RStudio', 'Visual Studio', 'Visual Studio Code', 'PyCharm', 'Spyder', 'Notepad++', 'Sublime Text',
    'Vim / Emacs', 'MATLAB',
    'None', 'Other'
]

plot_multi('Q9', 'Programming IDEs', names, len(names)-1)
```

</p></details>


![Programming IDEs](/images/output_13_0.png)


## Business Intelligence
This question was answered only by large enterprise employees. The top two picks both have a free desktop version, which is a major plus point because it means that employees have the option to stay current on these tools if they use the tools for their own personal projects outside of work (e.g. finance tracking). Google Data Studio only has a cloud version, while Qlik is not free to download (huge minus). It's interesting to see that "Others" came up 5th, but to understand what these tools are will require a more detailed survey.

For now, we stick to the top 3: **(1) Tableau, (2) Power BI, and (3) Google Data Studio**. 

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'Amazon QuickSight', 'Microsoft Power BI', 'Google Data Studio',
    'Looker', 'Tableau', 'Salesforce', 'Einstein Analytics', 'Qlik',
    'Domo', 'TIBCO Spotfire', 'Alteryx', 'Sisense', 'SAP Analytics Cloud',
    'None', 'Other'
]

plot_multi('Q31_A', 'BI Tools', names, len(names)-1)
```

</p></details>

</p></details>


![Business Intelligence Tools](/images/output_15_0.png)


## Data Visualisation (Dataviz) Libraries
Once again, respondents from both groups were in agreement on the top 4 Dataviz libraries. While the students preferred Geoplotlib for their 5th choice which is a strange because it's primarily for geo data, the large enterprise employees chose Shiny, which is also a strange choice because it's primarily for interactive web apps. Perhaps people from the two groups were working together to create COVID-19 dashboards this year.

The consensus top 3 + 1 (because I'm a huge fan of Plotly): **(1) Matplotlib, (2) Seaborn, (3) ggplot2, and (4) Plotly**.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'Matplotlib', 'Seaborn', 'Plotly', 'ggplot2', 'Shiny', 'D3.js',
    'Altair', 'Bokeh', 'Geoplotlib', 'Leaflet / Folium', 'None', 'Other'
]

plot_multi('Q14', 'Dataviz Libraries', names, len(names)-1)
```

</p></details>


![Data Visualisation Libraries](/images/output_17_0.svg)


## ML Frameworks
Again, the two groups were in agreement on the top 3. The students preferred PyTorch over XGBoost for their 4th and 5th choice respectively, while the large enterprise employees preferred the reverse. I was surprised that Spark MLlib wasn't in the mix. It is incredibly good for large datasets.

I wouldn't limit myself to any top *X* here, because it's good to test out as many open source frameworks as possible to see what works. However, if we had to choose an "enterprise" ML framework for standardisation, we should go for **scikit-learn for Python and Caret for R**.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Fast.ai',
    'MXNet', 'Xgboost', 'LightGBM', 'CatBoost', 'Prophet', 'H20', 'Caret',
    'Tidymodels', 'JAX', 'None', 'Other'
]

plot_multi('Q16', 'ML Frameworks', names, len(names)-1)
```

</p></details>


![ML Frameworks](/images/output_19_0.svg)


## Big Data Products
For this and subsequent questions, only large enterprise employees provided answers.

Strangely, Kaggle lumped "relational databases, data warehouses, data lakes, and similar tools" under the big data products category. In the [2019 Survey](https://www.kaggle.com/c/kaggle-survey-2019/), this was the categorisation used (changes in the 2020 survey are struck through):

* Big Data Products:
    * Google BigQuery
    * AWS Redshift
    * ~~Databricks~~
    * ~~AWS Elastic MapReduce~~
    * ~~Teradata~~
    * ~~Microsoft Analysis Services~~
    * ~~Google Cloud Dataflow~~
    * AWS Athena
    * ~~AWS Kinesis~~
    * ~~Google Cloud Pub/Sub~~
* Relational Database Products
    * MySQL
    * PostgresSQL
    * SQLite
    * Microsoft SQL Server
    * Oracle Database
    * Microsoft Access
    * ~~AWS Relational Database Service~~
    * AWS DynamoDB
    * ~~Azure SQL Database~~
    * Google Cloud SQL
    
The new tools were:

* MongoDB
* Snowflake
* IBM DB2
* Microsoft Azure Data Lake Storage
* Google Cloud Firestore

Using the 2019 categories, the top tools were:

| Rank | Big Data Products | Databases            |
| :--: | :---------------: | :------------------: |
| 1    | Google Cloud BigQuery | MySQL                |
| 2    | Azure Data Lake Storage | Microsoft SQL Server |
| 3    | AWS Redshift      | PostgreSQL           |
| 4    | AWS Athena        | Oracle Database      |
| 5    | -                 | MongoDB              |

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'MySQL', 'PostgreSQL', 'SQLite', 'Oracle Database', 'MongoDB',
    'Snowflake', 'IBM DB2', 'Microsoft SQL Server', 'Microsoft Access',
    'Microsoft Azure Data Lake Storage', 'AWS Redshift', 'AWS Athena',
    'Amazon DynamoDB', 'Google Cloud BigQuery', 'Google Cloud SQL', 'Google Cloud Firestore',
    'None', 'Other'
]

plot_multi('Q29_A', 'Big Data Products Used', names, len(names)-1)
```

</p></details>


![Big Data Products](/images/output_21_0.png)


## Automated ML Tools
I'm not yet sold on automated ML. The gist of my discomfort is that while you gain efficiency and democratise model development, you lose the expertise from developing models and you get a bunch of models that are weighted the same, regardless of whether it was developed by a PhD or that guy from marketing. I'll save this discussion for another post.

My point here is that I'm not familiar with automated ML. And so, I would have to go with the data here: **(1) Auto-SKlearn, (2) Auto-Keras, and (3) AutoML**.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'Google Cloud AutoML', 'H20 Driverless AI', 'Databricks AutoML',
    'DataRobot AutoML', 'Tpot', 'Auto-Keras', 'Auto-Sklearn', 'Auto_ml',
    'Xcessiv', 'MLbox', 'No / None', 'Other'
]
plot_multi('Q34_A', 'Automated ML Tools', names, len(names)-1)
```

</p></details>

![Automated ML Tools](/images/output_23_0.png)


## ML Management Tools
It's interesting that "Others" came up third. This suggests that perhaps, some companies develop their own ML management tools. Hence, we'll break the top picks into open source and paid ones. The open source solutions: **(1) TensorBoard, (2) Trains, (3) Sacred + Omniboard**. The paid, cloud-enabled solutions: **(1) Neptune.ai, (2) Weights & Biases, and (3) Comet.ml**.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'Neptune.ai', 'Weights & Biases', 'Comet.ml', 'Sacred + Omniboard',
    'TensorBoard', 'Guild.ai', 'Polyaxon', 'Trains',
    'Domino Model Monitor', 'No / None', 'Other'
]
plot_multi('Q35_A', 'ML Management Tools', names, len(names)-1)
```

</p></details>


![ML Management Tools](/images/output_25_0.png)


## Sharing & Deployment Tools
I don't think this question was sufficiently pointed. It actually asks about a few categories of tools. *GitHub* allows you to share code and perform version control, *Kaggle*, *Colab*, and *NBViewer* allow for sharing of notebooks online. *Shiny*, *Dash*, and *Streamlit* are lightweight web app frameworks. For version control and sharing of notebooks, I'm of the view that **(1) GitHub**, which has NBViewer built in, is sufficient. Then, I'd pick all three web app frameworks: **(2) Shiny, (3) Streamlit, and (4) Dash**. Shiny is hands-down the best web app framework for R, while Streamlit and Dash (for Python) have their own strengths.

<details><summary style="color: #8497B0;"><em>Code</em></summary>
<p>

```python
names = [
    'Plotly Dash', 'Streamlit', 'NBViewer', 'GitHub', 'Personal Blog',
    'Kaggle', 'Colab', 'Shiny', "None / Don't Share Publicly", 'Other'
]
plot_multi('Q36', 'Sharing & Deployment Tools', names, len(names)-1)
```

</p></details>


![Sharing & Deployment Tools](/images/output_27_0.png)


# Summary
To summarise, we picked out the following tools for large enterprises:

| Type of Tool | Top Picks |
| :----------- | :-------- |
| Programming Languages | Python, SQL, and R |
| IDEs | Jupyter, Visual Studio Code, PyCharm, and RStudio |
| Business Intelligence | Tableau, Power BI, and Google Data Studio |
| Data Visualisation Libraries | Matplotlib, Seaborn, ggplot2, Plotly |
| ML Frameworks | As many as possible; Default: Scikit-Learn for Python, Caret for R |
| Big Data Products (Cloud Only) | Google Cloud BigQuery, Azure Data Lake Storage, and AWS Redshift |
| Databases | MySQL, Microsoft SQL Server, PostgreSQL, and MongoDB |
| Automated ML | Auto-SKlearn, Auto-Keras, and AutoML |
| ML Management (Open Source) | TensorBoard, Trains, and Sacred + Omniboard |
| ML Management (Paid; Cloud) | Neptune.ai, Weights & Biases, and Comet.ml |
| Sharing & Deployment | Shiny, Streamlit, and Dash |

Many of these are open source, which means there is flexibility for large enterprises to configure them to fit their purposes. Of course, there may be other tools or entire categories of tools that we may be missing out here, but insights on what and how useful those tools are cannot be gleaned from the data available.
