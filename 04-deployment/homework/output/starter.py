#!/usr/bin/env python
# coding: utf-8

# In[19]:


#get_ipython().system('pip freeze | grep scikit-learn')


# In[20]:


#get_ipython().system('python -V')


# In[21]:


import pickle
import pandas as pd
import sys



# In[22]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[23]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[18]:
year = int(sys.argv[1])
month = int(sys.argv[2])

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[7]:


#pip install pyarrow


# In[8]:


#pip install --upgrade pip


# In[9]:


df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[10]:


df.head()


# In[11]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[12]:


print(y_pred)
print('mean predicted duration:',y_pred.mean())


# In[13]:


#std_dev = y_pred.std()
#print(std_dev)


# In[14]:


#df_result = pd.DataFrame()
#df_result['ride_id'] = df['ride_id']


# In[15]:


#df_result['predicted_duration'] = y_pred


# In[16]:


#df_result.to_parquet(
    #output_file,
    #engine='pyarrow',
    #compression=None,
    #index=False
#)


# In[17]:


#get_ipython().system('ls -ltrh output/')

