import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

plate = 'BR00113818_FS'

csv_file = f'/Users/rdijk/Documents/Data/ProcessedData/Stain2/{plate}.csv'
parquet_file = csv_file[:-3]+'parquet'
chunksize = 150_000

csv_stream = pd.read_csv(csv_file,
                         dtype={'broad_sample': str,
                         'pert_iname': str,
                         'moa': str,
                         #'gene': str,
                         'control_type': str},
                         chunksize=chunksize,
                         low_memory=False
                         )

for i, chunk in enumerate(csv_stream):
    print("Chunk", i)
    if i == 0:
        # Guess the schema of the CSV file from the first chunk
        parquet_schema = pa.Table.from_pandas(df=chunk).schema
        # Open a Parquet file for writing
        parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')
    # Write CSV chunk to the parquet file
    table = pa.Table.from_pandas(chunk, schema=parquet_schema)
    parquet_writer.write_table(table)

parquet_writer.close()

print('Donezo')


# param = 'Nuclei'
# FeatureNames = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/Stain2FeatureNames.csv')
# a = [c for c in FeatureNames.iloc[:,0] if c.startswith(param)]
# [print(f'{param}.'+x+',') for x in a]


#%%

# Convert csv files to parquet efficiently

import pandas as pd
import sqlite3
import os

# Metadata information
# metadata_path = '/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv'
# tablename = 'JUMP_MOA_compound_platemap_with_metadata_csv'

# Batch information
# path = r'/Users/rdijk/Documents/Data/RawData/Stain2/sqlite'
# batch = 'BR00112197binned.sqlite'

# Output file name
# parquet_file = os.path.join('/Users/rdijk/Documents/Data/ProcessedData/Stain2', batch[:-6]+'parquet')

# Sqlite files to be executed
# sqlfile1 = '/Users/rdijk/Library/DBeaverData/workspace6/Feature Aggregation/Scripts/Pipeline/01_CreateWellColumnImage.sql'
# sqlfile2 = '/Users/rdijk/Library/DBeaverData/workspace6/Feature Aggregation/Scripts/Scripts/PythonSql/PythonSql1.sql'
# sqlfile2 = '/Users/rdijk/Library/DBeaverData/workspace6/Feature Aggregation/Scripts/Pipeline/02_Stain2_FS_Agg.sql'
# sqlfile3 = '/Users/rdijk/Library/DBeaverData/workspace6/Feature Aggregation/Scripts/Scripts/PythonSql/PythonSql2.sql'
# sqlfile4 = '/Users/rdijk/Library/DBeaverData/workspace6/Feature Aggregation/Scripts/Scripts/PythonSql/PythonSql3.sql'

# Establish connection
# connection = sqlite3.connect(os.path.join(path, batch))
# cursor = connection.cursor()

# Import metadata table for joining of tables
# metadata_df = pd.read_csv(metadata_path)
# metadata_df.to_sql(tablename, connection, if_exists='append', index=False)

# Check if correctly imported
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())

# print('Creating well column...')
# sql_file = open(sqlfile1)
# sql_as_string = sql_file.read()
# cursor.executescript(sql_as_string)

# print('Joining tables and reading as pandas DF...')
# sql_file = open(sqlfile2)
# sql_as_string = sql_file.read()
# pandasDF = pd.read_sql_query(sql_as_string, connection)
# print('Writing DF as parquet file')
# pandasDF.to_parquet(parquet_file)