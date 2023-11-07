import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

plate = 'SQ00015223'

#csv_file = f'/Users/rdijk/Documents/Data/ProcessedData/Stain3/{plate}.csv'
csv_file = fr'/Users/rdijk/Documents/ProjectFA/Phase2/Data/ProcessedData/{plate}.csv'

parquet_file = csv_file[:-3] + 'parquet'
chunksize = 150_000

columns = pd.read_csv(csv_file, nrows=1).columns.tolist()
columns_dict = {sub: np.float32 for sub in columns}
columns_dict.update({'well_position': str,
                     'broad_sample': str,
                     'pert_iname': str,
                     'pert_type': str,
                     'control_type': str,
                     'moa': str,
                     'plate_map_name': str
                     })
csv_stream = pd.read_csv(csv_file,
                         dtype=columns_dict,
                         chunksize=chunksize,
                         low_memory=False,
                         on_bad_lines='skip',
                         na_values=['0.0077482126.0', '104.230.06618817979624629', '0.0422404840.9948853419886684',
                                    '0.2126100.0', '0..03280088609347405']
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
