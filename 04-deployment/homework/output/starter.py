#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')



# In[8]:


pip install pyarrow


# In[9]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[14]:


df.head()


# In[10]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[12]:


print(y_pred)


# In[13]:


std_dev = y_pred.std()
print(std_dev)


# In[15]:


year = 2023
month = 3

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
df = read_data(input_file)


# In[21]:


output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[16]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[17]:


df.head()


# In[18]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']


# In[19]:


df_result['predicted_duration'] = y_pred


# In[23]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[28]:


get_ipython().system('ls -ltrh output/')


# In[29]:


pip install nbconvert


# In[30]:


pip install --upgrade pip


# In[31]:


pip install nbconvert


# In[44]:


#cd /workspaces/mlops-zoomcamp0o4-deployment/homework/output
jupyter nbconvert --to script --output-dir=/workspaces/mlops-zoomcamp/0o4-deployment/homework/output starter.ipynb

