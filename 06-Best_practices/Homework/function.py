#def read_data(categorical):
#import all modules
#cpoy read function,add categorical variable
#add df[ride] column
#read that model
#transform,predict 

import pickle
import pandas as pd
import sys
def main(year,month):
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    categorical = ['PULocationID', 'DOLocationID']
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    def read_data(input_file,categorical):

       df = pd.read_parquet(input_file)
    
       df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
       df['duration'] = df.duration.dt.total_seconds() / 60

       df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
       categorical = ['PULocationID', 'DOLocationID']
       df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
       df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
       #return df
       dicts = df[categorical].to_dict(orient='records')
       with open('model.bin', 'rb') as f_in:
          dv, model = pickle.load(f_in)

       #dicts = df[categorical].to_dict(orient='records')
       X_val = dv.transform(dicts)
       y_pred = model.predict(X_val)
       df_result = pd.DataFrame()
       df_result['ride_id'] = df['ride_id']
       df_result.to_parquet(output_file,engine='pyarrow',compression=None,index=False)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year,month)

