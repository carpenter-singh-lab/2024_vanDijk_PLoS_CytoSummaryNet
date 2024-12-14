import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.preprocessing import StandardScaler

# Training stuff
from pytorch_metric_learning import losses, distances
from networks.SimpleMLPs import MLPsumV2
import torch.optim as optim
import torch
from tqdm import tqdm

# Evaluation stuff
import pandas as pd
import utils
import copy


#%%
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_correlated_dataset(n, dependency, mu, input_dim):
    latent = np.random.randn(n, input_dim)
    dependent = latent.dot(dependency)
    scaled = dependent
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

#%%
if __name__ == "__main__":

    train_model = False  # TODO hyperparameter
    load_pretrained_model = True  # TODO hyperparameter
    plot_distributions = True  # TODO hyperparameter
    standardize = False  # TODO hyperparameter
    spherize = False  # TODO hyperparameter

    input_dim = 2
    fill_to_model_input_dim = True
    roll_dims = True

    ##% End hyperparameter selection
    if spherize:
        regularization = 0.01  # [0.01, 0.1
        rv = 0.99

    np.random.seed(42)

    n_classes = 10
    mean = [0, 0]
    cov = np.array([[1, 0.5], [0.5, 1]]) # default covariance matrix for correct ratios

    ELLIPSOIDS = {}
    # samples = []
    cov_angle = np.linspace(0, np.pi, 11)
    for i in range(n_classes):
        t = cov_angle[i]  # between 0 and pi
        rotation_matrix = np.array([[np.cos(t), -np.sin(t)],
                           [np.sin(t), np.cos(t)]])

        ellipsoid_cov = rotation_matrix @ cov @ rotation_matrix.T

        ELLIPSOIDS[str(i)] = ellipsoid_cov

    ##% Plotting the distributions
    if plot_distributions:
        L = 0
        fig, axs = plt.subplots(2, 5, figsize=(9, 3), dpi=300)
        for ax, (title, dependency) in zip(axs.flat, ELLIPSOIDS.items()):
            x, y = get_correlated_dataset(800, dependency, mean, input_dim)
            data = np.column_stack([x, y])
            if standardize:
                scaler = StandardScaler(copy=False).fit(data)
                data = scaler.transform(data)
                x, y = data[:, 0], data[:, 1]

            if spherize:
                ZCA = utils.ZCA(regularization=regularization, retain_variance=rv).fit(data)
                data = ZCA.transform(data)
                x, y = data[:, 0], data[:, 1]
            ax.scatter(x, y, s=0.5)

            ax.axvline(c='grey', lw=1)
            ax.axhline(c='grey', lw=1)

            confidence_ellipse(x, y, ax, edgecolor='red')

            ax.scatter(mean[0], mean[1], c='red', s=3)
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(L))
            L += 1
        plt.show()


    ##% GET DATA FOR TRAINING
    nr_samples_per_class = 4
    nr_points_per_sample = 800

    if load_pretrained_model:
        model = MLPsumV2(input_dim=1324, latent_dim=2048, output_dim=2048,
                         k=0.5, dropout=0, cell_layers=1,
                         proj_layers=2, reduction='sum')
        run_name = r'run-20220517_164505-2q0md5h8'
        print(f'Resuming training with {run_name}...')
        checkpoint = torch.load(f'wandb/{run_name}/files/model_bestval_simpleMLP_V1')
        model.load_state_dict(checkpoint)

    if roll_dims:
        apmlp = []
        apbm = []
        for roll_nr in tqdm(range(1324-2)):
            train_samples = []
            val_samples = []
            label_idx = 0
            for i, dependency in ELLIPSOIDS.items():
                x, y = get_correlated_dataset(nr_points_per_sample*nr_samples_per_class, dependency, mean, input_dim)
                data = np.column_stack([x, y])

                if standardize:
                    scaler = StandardScaler(copy=False).fit(data)
                    data = scaler.transform(data)
                    x, y = data[:, 0], data[:, 1]
                if spherize:
                    ZCA = utils.ZCA(regularization=regularization, retain_variance=rv).fit(data)
                    data = ZCA.transform(data)
                    x, y = data[:, 0], data[:, 1]

                for index in range(0, len(x), nr_points_per_sample):
                    x_set, y_set = x[index: index+nr_points_per_sample], y[index: index+nr_points_per_sample]
                #if int(i) == 0 or int(i) == 4 or int(i) == 5 or int(i) == 7:
                    if fill_to_model_input_dim:
                        s = np.column_stack([x_set, y_set, np.zeros((x_set.shape[0], 1324-input_dim))])
                        s = np.roll(s, roll_nr)
                        val_samples.append({'data': s,
                                              'cov_matrix': dependency,
                                              'label': label_idx})

                    else:
                        val_samples.append({'data': np.column_stack([x_set, y_set]),
                             'cov_matrix': dependency,
                             'label': label_idx})
                #else:
                    if fill_to_model_input_dim:
                        s = np.column_stack([x_set, y_set, np.zeros((x_set.shape[0], 1324 - input_dim))])
                        s = np.roll(s, roll_nr)
                        train_samples.append({'data': s,
                                              'cov_matrix': dependency,
                                              'label': label_idx})
                    else:
                        train_samples.append({'data': np.column_stack([x_set, y_set]),
                             'cov_matrix': dependency,
                             'label': label_idx})
                label_idx += 1


            ##% TRAINING
            if train_model:
                model = MLPsumV2(input_dim=input_dim, latent_dim=64, output_dim=2,  # latent_dim=64
                                 k=4, dropout=0, cell_layers=1,  # k=4
                                 proj_layers=2, reduction='sum')
                optimizer = optim.AdamW(model.parameters(), lr=1e-5)
                loss_func = losses.SupConLoss(distance=distances.CosineSimilarity())

                epochs = 10000
                bs = len(train_samples)//4

                #for e in tqdm(range(epochs)):
                for e in range(epochs):
                    model.train()
                    tr_loss = 0.0
                    for idx in range(0, len(train_samples), bs):
                        points = torch.stack([torch.tensor(x['data'], dtype=torch.float32) for x in train_samples[idx:idx + bs]])
                        labels = torch.stack([torch.tensor(x['label'], dtype=torch.int16) for x in train_samples[idx:idx + bs]])

                        feats, _ = model(points)

                        tr_loss_tmp = loss_func(feats, labels)
                        tr_loss += tr_loss_tmp.item()

                        tr_loss_tmp.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    tr_loss /= (idx + 1)


                    # Validation
                    model.eval()
                    best_val = np.inf
                    val_loss = 0.0
                    with torch.no_grad():
                        for idx in range(0, len(val_samples), bs):
                            points = torch.stack([torch.tensor(x['data'], dtype=torch.float32) for x in val_samples[idx:idx + bs]])
                            labels = torch.stack([torch.tensor(x['label'], dtype=torch.int16) for x in val_samples[idx:idx + bs]])
                            feats, _ = model(points)

                            val_loss_tmp = loss_func(feats, labels)
                            val_loss += val_loss_tmp.item()

                    if val_loss < best_val:
                        best_val = val_loss
                        best_model = copy.deepcopy(model)

                    print(f"Epoch {e}. Training loss: {tr_loss}. Validation loss: {val_loss}.")

            #%% Evaluate model performance
            model.eval()
            MLP_profiles = pd.DataFrame()
            BM_profiles = pd.DataFrame()
            with torch.no_grad():
                for idx in range(len(val_samples)):
                    points = torch.unsqueeze(torch.tensor(val_samples[idx]['data'], dtype=torch.float32), dim=0)
                    labels = val_samples[idx]['label']
                    feats, _ = model(points)

                    # Append everything to dataframes
                    c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
                    MLP_profiles = pd.concat([MLP_profiles, c1])

                    c2 = pd.concat([pd.DataFrame(points.mean(dim=1)), pd.Series(labels)], axis=1)
                    BM_profiles = pd.concat([BM_profiles, c2])

            MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
            BM_profiles.columns = [f"f{x}" for x in range(BM_profiles.shape[1] - 1)] + ['Metadata_labels']
            AP_MLP = utils.CalculateMAP(MLP_profiles, 'cosine_similarity', groupby='Metadata_labels')
            apmlp.append(AP_MLP.AP.mean())
            # print('Total model mAP:', AP_MLP.AP.mean(), '\nTotal model precision at R:', AP_MLP['precision at R'].mean())
            # print(AP_MLP.groupby(by='compound').mean().sort_values(by='AP', ascending=False).to_markdown())

            AP_BM = utils.CalculateMAP(BM_profiles, 'cosine_similarity', groupby='Metadata_labels')
            apbm.append(AP_BM.AP.mean())
            # print('Total baseline (mean) mAP:', AP_BM.AP.mean(), '\nTotal baseline (mean) precision at R:', AP_BM['precision at R'].mean())
            # print(AP_BM.groupby(by='compound').mean().sort_values(by='AP', ascending=False).to_markdown())

    plt.figure()
    plt.scatter(list(range(1322)), apbm, label='baseline')
    plt.scatter(list(range(1322)), apmlp, label='model')
    plt.legend()
    plt.ylabel('mAP')
    plt.xlabel('feature position')
    plt.show()

    cfnames = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CommonFeatureNames.csv', index_col=False)['FeatureNames']
    cfnames1 = np.array([x.split('.')[-1].split('_')[1] for x in cfnames])
    cfnames2 = np.array([x.split('.')[-1].split('_')[2] for x in cfnames])
    cfnames = np.where(cfnames2=='IntegratedIntensity', 1, 0)
    # cfnames = np.where(np.logical_or(np.logical_or(np.logical_or(cfnames1=='AreaShape', cfnames1=='Neighbors'),
    #                 cfnames2=='IntegratedIntensity'), cfnames2=='IntegratedIntensityEdge'), 1, 0)
    cfnames = cfnames[1:-1]

    fig, ax = plt.subplots(1,1)
    scatter = ax.scatter(list(range(1322)), apmlp, c=cfnames)
    ax.set_ylabel('mAP')
    ax.set_xlabel('feature position')
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()
