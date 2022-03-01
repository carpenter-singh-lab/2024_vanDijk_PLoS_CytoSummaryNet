import pandas as pd
import os
import matplotlib.pyplot as plt
import umap.plot

rootDir_MLP = r'/Users/rdijk/PycharmProjects/featureAggregation/outputs'
rootdir_BM = r'/Users/rdijk/Documents/Data/ProcessedData/Stain2/profiles'
MLP_platedirs = [x[2] for x in os.walk(rootDir_MLP)][0][1:]
MLP_platenames = [x.split('_')[-1] for x in MLP_platedirs]

BM_platedirs = [x[2] for x in os.walk(rootdir_BM)][0]
BM_platedirs = [x for x in BM_platedirs if x in MLP_platenames]

MLPdfs = []
BMdfs = []
i = 0
negconsStain = ["A11", "B18", "D17", "D19", "E07", "E08", "F07", "F24", "G20", "G23", "H10", "I12", "I13", "J01", "J21",
                "K24", "M05", "M06", "N06", "N22", "O14", "O19", "P02", "P11"]
for MLP, BM in zip(MLP_platedirs, BM_platedirs):
    temp1 = pd.concat([pd.read_csv(os.path.join(rootDir_MLP, MLP), index_col=False), pd.Series([i] * 360)], axis=1)
    MLPdfs.append(temp1)
    temp2 = pd.concat([pd.read_csv(os.path.join(rootdir_BM, BM), index_col=False), pd.Series([i] * 384)], axis=1)
    temp2 = temp2[~temp2.Metadata_Well.isin(negconsStain)] # remove negcons
    BMdfs.append(temp2)
    i += 1

# %% Create UMap
plt.figure(figsize=(14, 10), dpi=300)
BMDF = pd.concat([x for x in BMdfs[:-1]])
featuresBM = BMDF.iloc[:, 2:-1]
labels = BMDF.iloc[:, 0]
reducer = umap.UMAP()
embedding1 = reducer.fit(featuresBM)
umap.plot.points(embedding1, labels=labels, theme='fire')
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP MLP')
plt.show()

plt.figure(figsize=(14, 10), dpi=300)
MLPDF = pd.concat([x for x in MLPdfs[:-1]])
featuresMLP = MLPDF.iloc[:, :-2]
labels = BMDF.iloc[:, 0]
reducer = umap.UMAP()
embedding1 = reducer.fit(featuresMLP)
umap.plot.points(embedding1, labels=labels, theme='fire')
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP BM')
plt.show()