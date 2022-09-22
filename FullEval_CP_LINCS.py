# Standard libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Seeds
import random

# PyTorch
import torch
import torch.utils.data as data

# Custom libraries
from networks.SimpleMLPs import MLPsumV2
from dataloader_pickles import DataloaderEvalV5
import utils

# Average profiles preprocessing
from pycytominer.operations.transform import RobustMAD
from pycytominer import feature_select

# Statistics
import scipy.stats as stats

# Argument parser
import argparse


def fulleval(args):
    ##
    NUM_WORKERS = 0
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)

    # Set random seed for reproducibility
    manualSeed = 42
    print("Random Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    ## Load model
    save_name_extension = 'model_bestval_simpleMLP_V1'  # extension of the saved model
    model_name = save_name_extension
    print('Loading:', model_name)

    input_dim = args.model_input_size
    kFilters = args.kfilters
    latent_dim = 2048
    output_dim = 2048
    model = MLPsumV2(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                     k=kFilters, dropout=0, cell_layers=1,
                     proj_layers=2, reduction='sum')
    if torch.cuda.is_available():
        model.cuda()

    save_features_to_csv = args.save_csv_features
    train_val_split = 1.0

    percent_matching = args.find_sister_compounds  # If false calculate percent replicating

    if percent_matching:
        encoding_label = 'moa'
        mAP_label = 'Metadata_moa'
        bestMOAs = pd.DataFrame()
    else:
        encoding_label = 'pert_iname'
        mAP_label = 'Metadata_pert_iname'

    dataset_name = args.dataset_name
    MAPfilename = f'mAP_{dataset_name}_{args.dose_point}'

    path = args.model_path
    fullpath = os.path.join(path, 'files', model_name)

    if 'ckpt' in model_name:
        model.load_state_dict(torch.load(fullpath)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(fullpath))
    model.eval()


    ## Load all data
    rootDir = fr'datasets/{dataset_name}'
    plateDirs = [x[0] for x in os.walk(rootDir)][1:]
    platenames = [x.split('_')[-1] for x in plateDirs]

    metadata_dir = args.metadata_path  # path to metadata
    barcode_platemap = pd.read_csv(os.path.join(metadata_dir, 'barcode_platemap.csv'), index_col=False)
    barcode_platemap = barcode_platemap[barcode_platemap['Assay_Plate_Barcode'].isin(platenames)]

    repurposing_info = pd.read_csv(os.path.join(metadata_dir, 'repurposing_info_long.tsv'), index_col=False,
                                   low_memory=False, sep='\t', usecols=["broad_id", "pert_iname", "moa"])
    repurposing_info = repurposing_info.rename(columns={"broad_id": "broad_sample"})
    repurposing_info = repurposing_info.drop_duplicates()

    platemaps = barcode_platemap['Plate_Map_Name'].tolist()
    platenames = barcode_platemap['Assay_Plate_Barcode'].tolist()

    plateDirs = ['DataLoader_'+x for x in platenames]

    I = [i for i, y in enumerate(platemaps) if y == "C-7161-01-LM6-013" or y == "C-7161-01-LM6-001"]
    for ele in sorted(I, reverse=True):
        del plateDirs[ele]
        del platemaps[ele]
        del platenames[ele]

    assert len(plateDirs) == len(platenames) == len(platemaps)

    # Initialize variables
    average_perturbation_map = {}

    bigdf = []
    for i, pDir in enumerate(plateDirs):
        platestring = pDir.split('_')[-1]
        print('Getting data from: ' + platestring)
        C_plate_map = pd.read_csv(os.path.join(metadata_dir, 'platemap', platemaps[i]+'.txt'), sep='\t')
        C_metadata = utils.addDataPathsToMetadata(rootDir, C_plate_map, pDir)
        if args.dose_point == '10':
            df = C_metadata[np.logical_and(C_metadata['mmoles_per_liter'] > 9, C_metadata['mmoles_per_liter'] < 11)]
        elif args.dose_point == '3':
            df = C_metadata[np.logical_and(C_metadata['mmoles_per_liter'] > 2.9, C_metadata['mmoles_per_liter'] < 6)]
        bigdf.append(df)

    bigdf = pd.merge(pd.concat(bigdf), repurposing_info, on='broad_sample', how='left')
    bigdf = utils.filterData(bigdf, 'negcon', encode='pert_iname', mode='LINCS')
    shape1 = bigdf.shape[0]
    bigdf.dropna(inplace=True)  # drop all compounds without annotations for pert_iname (and moa)
    shape2 = bigdf.shape[0]
    print("Removed", shape1-shape2, "wells due to missing annotation of pert_iname and moa.")
    bigdf = bigdf[bigdf.Metadata_labels.duplicated(keep=False)]
    shape3 = bigdf.shape[0]
    print("Removed", shape2-shape3, "unique compound wells.")
    print('Using', shape3, "wells")

    Total, _ = utils.train_val_split(bigdf, 1.0, sort=True)
    ValDataset = DataloaderEvalV5(Total)
    loader = data.DataLoader(ValDataset, batch_size=1, shuffle=False,
                                   drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)


    ## Create feature dataframes
    MLP_profiles = pd.DataFrame()
    average_profiles = pd.DataFrame()

    print('Calculating Features')

    with torch.no_grad():
        for (points, labels) in tqdm(loader):
            if points.shape[1] == 1:
                continue
                #points = torch.zeros(1, 1, args.model_input_size)

            feats, _ = model(points)
            # Append everything to dataframes
            c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
            MLP_profiles = pd.concat([MLP_profiles, c1])

            # Calculate benchmark
            c2 = pd.concat([pd.DataFrame(points.mean(dim=1)), pd.Series(labels)], axis=1)
            average_profiles = pd.concat([average_profiles, c2])

    ## Rename columns and normalize features
    # Rename feature columns
    MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
    average_profiles.columns = [f"Cells_{x}" for x in range(average_profiles.shape[1] - 1)] + ['Metadata_labels']

    MLP_profiles = MLP_profiles[MLP_profiles.Metadata_labels.duplicated(keep=False)]  # filter out non-replicates
    average_profiles = average_profiles[average_profiles.Metadata_labels.duplicated(keep=False)]  # filter out non-replicates
    MLP_profiles.reset_index(drop=True, inplace=True)
    average_profiles.reset_index(drop=True, inplace=True)

    print('MLP_profiles shape: ', MLP_profiles.shape)
    print('average_profiles shape: ', average_profiles.shape)

    ## Preprocess the average profiles
    scaler = RobustMAD(epsilon=0)
    fitted_scaler = scaler.fit(average_profiles.iloc[:, :-1])
    features = fitted_scaler.transform(average_profiles.iloc[:, :-1])
    features = feature_select(average_profiles.iloc[:, :-1], operation=["variance_threshold",
                                                                         "correlation_threshold",
                                                                         "drop_na_columns",
                                                                         "blocklist"])
    average_profiles = pd.concat([features, average_profiles.iloc[:, -1]], axis=1)

    ## Save all the dataframes to .csv files!
    try:
        os.mkdir(args.output_path)
    except:
        pass

    try:
        os.mkdir(f'{args.output_path}/{dataset_name}_profiles')
    except:
        pass


    if save_features_to_csv:
        MLP_profiles.to_csv(f'{args.output_path}/{dataset_name}_profiles/MLP_profiles_{args.dose_point}_.csv', index=False)
        average_profiles.to_csv(f'{args.output_path}/{dataset_name}_profiles/average_profiles_{args.dose_point}_.csv', index=False)
    split = int(MLP_profiles.shape[0] * train_val_split)

    temp_df = bigdf[['Metadata_labels', 'moa', 'pert_iname']][~bigdf.Metadata_labels.duplicated(keep='last')]
    temp_df = temp_df.rename(columns={'moa': 'Metadata_moa', 'pert_iname': 'Metadata_pert_iname'})
    MLP_profiles = pd.merge(MLP_profiles, temp_df, on='Metadata_labels')
    average_profiles = pd.merge(average_profiles, temp_df, on='Metadata_labels')

    print('Dropping ', MLP_profiles.shape[0] - MLP_profiles.dropna().reset_index(drop=True).shape[0], 'rows due to NaNs')
    MLP_profiles = MLP_profiles.dropna().reset_index(drop=True)
    average_profiles = average_profiles.dropna().reset_index(drop=True)
    print('New size:', MLP_profiles.shape)
    shuffled_profiles = average_profiles.copy()
    shuffled_profiles.iloc[:, :-3] = pd.DataFrame.from_records(np.ones(shuffled_profiles.iloc[:, :-3].shape))

    ## Calculate mean average precision
    ap_mlp = utils.CalculateMAP(MLP_profiles, 'cosine_similarity',
                            groupby=mAP_label, percent_matching=percent_matching)
    ap_bm = utils.CalculateMAP(average_profiles, 'cosine_similarity',
                            groupby=mAP_label, percent_matching=percent_matching)
    ap_shuffled = utils.CalculateMAP(shuffled_profiles, 'cosine_similarity',
                            groupby=mAP_label, percent_matching=percent_matching)

    print('Total mean mAP MLP:', ap_mlp.AP.mean(), '\nTotal mean precision at R MLP:', ap_mlp['precision at R'].mean())
    print(ap_mlp.sort_values('AP').iloc[-30:, :].round(4).to_markdown(index=False))
    print('\n')
    print('Total mean mAP BM:', ap_bm.AP.mean(), '\nTotal mean precision at R BM:', ap_bm['precision at R'].mean())
    print(ap_bm.sort_values('AP').iloc[-30:, :].round(4).to_markdown(index=False))
    print('\n')
    print('Total mean mAP shuffled:', ap_shuffled.AP.mean(), '\nTotal mean precision at R shuffled:', ap_shuffled['precision at R'].mean())

    print('\n')
    # Conduct Welch's t-Test and print the result
    print("Welch's t-test between mlp mAP and bm mAP:", stats.ttest_ind(np.array(ap_mlp.AP), np.array(ap_bm.AP), equal_var=False))
    print('\n')

    # WRITE TO FILE
    try:
        os.mkdir(f'{args.output_path}/mAPs')
    except:
        pass

    f = open(f'{args.output_path}/mAPs/{MAPfilename}.txt', 'a')
    f.write('\n')
    f.write('\n')
    f.write(f'Dataset: {args.dataset_name}')
    f.write('\n')
    f.write('\n')
    f.write('MLP results')
    f.write('\n')
    f.write('Total mean:' + str(ap_mlp.AP.mean()))
    f.write('\n')
    f.write(f'Training samples || mean:{ap_mlp.iloc[:split,1].mean()}\n' + ap_mlp.iloc[:split, :].groupby(
        'compound').mean().sort_values(by='AP',ascending=False).to_markdown())
    f.write('\n')
    f.write(f'Validation samples || mean:{ap_mlp.iloc[split:,1].mean()}\n' + ap_mlp.iloc[split:, :].groupby(
        'compound').mean().sort_values(by='AP',ascending=False).to_markdown())
    f.write('\n')
    f.write('\n')
    f.write('BM results')
    f.write('\n')
    f.write('Total mean:' + str(ap_bm.AP.mean()))
    f.write('\n')
    f.write(f'Training samples || mean:{ap_bm.iloc[:split, 1].mean()}\n' + ap_bm.iloc[:split, :].groupby(
        'compound').mean().sort_values(by='AP', ascending=False).to_markdown())
    f.write('\n')
    f.write(f'Validation samples || mean:{ap_bm.iloc[split:, 1].mean()}\n' + ap_bm.iloc[split:, :].groupby(
        'compound').mean().sort_values(by='AP', ascending=False).to_markdown())
    f.write('\n')
    f.write('\n')
    f.write('Shuffled results')
    f.write('\n')
    f.write('Total mean:' + str(ap_shuffled.AP.mean()))
    f.write('\n')
    f.write(f'Training samples || mean:{ap_shuffled.iloc[:split, 1].mean()}\n' + ap_shuffled.iloc[:split, :].groupby(
        'compound').mean().sort_values(by='AP', ascending=False).to_markdown())
    f.write('\n')
    f.write(f'Validation samples || mean:{ap_shuffled.iloc[split:, 1].mean()}\n' + ap_shuffled.iloc[split:, :].groupby(
        'compound').mean().sort_values(by='AP', ascending=False).to_markdown())
    f.close()

    # Add to a large dataframe
    if percent_matching:
        allresultsdf =pd.DataFrame({'mAP model': [ap_mlp.iloc[:, 1].mean()],
                                    'mAP BM': [ap_bm.iloc[:, 1].mean()],
                                    'mAP shuffled': [ap_shuffled.iloc[:, 1].mean()]})
        sorted_dictionary = {k: [v] for k, v in sorted(average_perturbation_map.items(), key=lambda item: item[1], reverse=True)}
        bestmoas = pd.DataFrame(sorted_dictionary)
    else:
        allresultsdf = pd.DataFrame({'Training mAP model': [ap_mlp.iloc[:split, 1].mean()],
                                     'Training mAP BM': [ap_bm.iloc[:split, 1].mean()],
                                     'Training mAP shuffled': [ap_shuffled.iloc[:split, 1].mean()],
                                     'Validation mAP model': [ap_mlp.iloc[split:, 1].mean()],
                                     'Validation mAP BM': [ap_bm.iloc[split:, 1].mean()],
                                     'Validation mAP shuffled': [ap_shuffled.iloc[split:, 1].mean()]})

    ## Organize all results and save them
    PLATES = [x.split('_')[-1] for x in plateDirs]
    PLATES.sort()
    allresultsdf['plate'] = '_'.join(PLATES)

    allresultsdf = allresultsdf.set_index('plate')
    print(allresultsdf.round(4).to_markdown())

    if percent_matching:
        allresultsdf.to_csv(f'{args.output_path}/{dataset_name}_moa_map.csv')
    else:
        allresultsdf.to_csv(f'{args.output_path}/{dataset_name}_replicating_map.csv')

    if percent_matching:
        bestmoas.to_csv(f'{args.output_path}/bestMoAs_{dataset_name}.csv')



## MAIN
if __name__=='__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description="Evaluate a trained model by calculating mAP for either"
                                                 "finding a compound's replicate or sister compound.",
                                     fromfile_prefix_chars='@')

    # Optional positional argument
    parser.add_argument('model_input_size', nargs='?', const=1781, type=int,
                        help='Number of single cell features.')
    # Optional positional argument
    parser.add_argument('kfilters', nargs='?', const=1/2, type=float,
                        help='Times division of the number of filters in the hidden model layers')
    # Optional positional argument
    parser.add_argument('save_csv_features', nargs='?', const=True, type=bool,
                        help='If True save the model aggregated profiles as a .csv file.')
    # Optional positional argument
    parser.add_argument('find_sister_compounds', nargs='?', const=False, type=bool,
                        help='If True calculate mAP for finding sister compounds, for compound replicates if False.')

    # Optional positional argument
    parser.add_argument('dataset_name', nargs='?', const='LINCS', type=str,
                        help='Dataset name')
    # Optional positional argument
    parser.add_argument('model_path', nargs='?', const='wandb/latest-run/files', type=str,
                        help='Number of epochs to train the model')
    # Optional positional argument
    parser.add_argument('metadata_path', nargs='?', const='aws_scripts/metadata/platemaps/2016_04_01_a549_48hr_batch1',
                        type=str, help='Number of epochs to train the model')
    # Optional positional argument
    parser.add_argument('dose_point', nargs='?', const='10', type=str,
                        help='Dose point for which to calculate the metrics. Either 10 or 3.33 uM')
    # Optional positional argument
    parser.add_argument('output_path', nargs='?', const='Results', type=str, help='Path to save all output files')

    # Parse arguments
    args = parser.parse_args()

    print("Argument values:")
    print('model input size:', args.model_input_size)
    print('kfilters:', args.kfilters)
    print('find sister compounds:', args.find_sister_compounds)
    print('dataset name:', args.dataset_name)
    print('model path:', args.model_path)
    print('metadata path:', args.metadata_path)
    print('dose point:', args.dose_point)
    print('output_path:', args.output_path)

    fulleval(args)

