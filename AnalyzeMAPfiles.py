
import pandas as pd
import io
import plotly.express as px


BM_map_path = '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MAPs/MAP_BM.txt'

MLP_map_path = '/Users/rdijk/PycharmProjects/featureAggregation/figures/Stain2/Stain2_wellAugment_noCP/MAP_wellAugment_noCP.txt'

compounds = 'training'

with open(BM_map_path) as file:
    BM_plates = []
    BM_trainingstarts = [] # 72 compounds
    BM_valstarts = [] # 18 compounds
    for i, line in enumerate(file):
        if line.startswith('Plate:'):
            BM_plates.append(line.split(' ')[-1][:-1])
        if line.startswith('Training samples'):
            BM_trainingstarts.append(i+2)
        if line.startswith('Validation samples'):
            BM_valstarts.append(i+2)

with open(MLP_map_path) as file:
    MLP_plates = []
    MLP_trainingstarts = [] # 72 compounds
    MLP_valstarts = [] # 18 compounds
    for i, line in enumerate(file):
        if line.startswith('Plate:'):
            MLP_plates.append(line.split(' ')[-1][:-1])
        if line.startswith('Training samples'):
            MLP_trainingstarts.append(i+1)
        if line.startswith('Validation samples'):
            MLP_valstarts.append(i+1)

if compounds == 'training':
    BM_starts = BM_trainingstarts
    MLP_starts = MLP_trainingstarts
    C = 74
elif compounds == 'validation':
    BM_starts = BM_valstarts
    MLP_starts = MLP_valstarts
    C = 20
else:
    raise Warning('incorrect compound category')



file = open(BM_map_path)
all_content_BM = file.readlines()
file.close()

BM_dict = {}
for j in range(len(BM_starts)):
    BM_content = all_content_BM[BM_starts[j]:BM_starts[j]+C]
    BM_content.pop(1)
    mdtable = pd.read_csv(io.StringIO(''.join(BM_content)), sep='|').iloc[:, 1:3]
    mdtable = mdtable.rename(columns={mdtable.columns[0]: 'compound', mdtable.columns[1]: 'AP'})
    BM_dict[BM_plates[j]] = mdtable

file = open(MLP_map_path)
all_content_MLP = file.readlines()
file.close()
MLP_dict = {}
for j in range(len(MLP_starts)):
    MLP_content = all_content_MLP[MLP_starts[j]:MLP_starts[j]+C]
    MLP_content.pop(1)
    mdtable = pd.read_csv(io.StringIO(''.join(MLP_content)), sep='|').iloc[:, 1:3]
    mdtable = mdtable.rename(columns={mdtable.columns[0]: 'compound', mdtable.columns[1]: 'AP'})
    MLP_dict[MLP_plates[j]] = mdtable


#%% Now let's plot
for k, plate in enumerate(MLP_plates):
    BM = BM_dict[plate]
    MLP = MLP_dict[plate]
    DF = pd.merge(BM, MLP, on='compound')
    DF['mAP diff'] = DF['AP_y'] - DF['AP_x']
    DF = DF.sort_values(by='compound')

    fig = px.scatter(DF, x='compound', y='mAP diff', hover_data=['compound'])
    fig.show()
    # fig = plt.figure(figsize=(10, 10), dpi=150)
    # plt.scatter()
    # plt.title(plate)
    # plt.ylim(-1, 1)
    # plt.ylabel('MLP mAP - BM mAP')
    # plt.show()


