# Introduction  

This repository contains the code to reproduce the model training, inference, and analysis presented in the paper:  
**"Capturing cell heterogeneity in representations of cell populations for image-based profiling using contrastive learning"**  
*(van Dijk, R., Arevalo, J., Babadi, B., Carpenter, A. E., & Singh, S., 2024, PLOS Computational Biology)*.  
[Read the full paper here](https://doi.org/10.1371/journal.pcbi.1012547).  

We propose a Deep Sets-based method that learns the optimal way of aggregating single-cell feature data into a profile, improving the prediction of a compound’s mechanism of action compared to traditional average profiling. This is achieved through weakly supervised contrastive learning in a multiple-instance learning setting. Our approach offers a more accessible and effective method for aggregating single-cell feature data than previous studies, significantly outperforming the average profiling baseline.  

This repository provides:  
- **`building_blocks/`**: Scripts for building your own pipeline for developing, training, and evaluating the model. We recommend using these if you plan on implementing CytoSummaryNet on your own dataset.
- **`src/`**: Scripts for reproducing the results for `cpg0001` and `cpg0004`.
- **`paper_figures/`**: Jupyter notebooks for generating the figures and reproducing the results presented in the paper.

---

# To Build Your Own Pipeline  

This repository includes modular components in the `building_blocks/` directory to help you implement your own version of CytoSummaryNet on your dataset. Below is an overview of the main components and where to get started:

## **Key Components**  

### 1. **Dataset Class and Collate Function**  
- **File**: `building_blocks/dataset.py`  
- **Description**: This file defines the PyTorch `Dataset` class and collate function for handling single-cell data. It includes data augmentation strategies, imbalanced label handling, and grouping functionality for efficient data loading.  
- **Getting Started**:  
  - Define your dataset by creating a DataFrame with file paths to pickle files and labels.  
  - Use the `TemplateDataset` class to load your data and perform data augmentation as needed.  
  - Mock tests for the dataset are available in `building_blocks/tests/test_dataset.py`.

### 2. **Model Architecture**  
- **File**: `building_blocks/models.py`  
- **Description**: This file implements CytoSummaryNet, a modular and parameterizable PyTorch model designed to aggregate single-cell features into population-level representations.  
- **Getting Started**:  
  - Use the `CytoSummaryNet` class to initialize your model.  
  - Customize input dimensions, layer configurations, and pooling strategies based on your dataset.  
  - Unit tests for the model architecture are available in `building_blocks/tests/test_model.py`.

### 3. **Training Loop**  
- **File**: `building_blocks/engine.py`  
- **Description**: This file defines the training loop, including loss computation, backpropagation, and validation. It is designed to integrate seamlessly with the dataset and model components.  
- **Getting Started**:  
  - Use the `train_loop` function to train your model.  
  - Provide it with your `DataLoader`, `CytoSummaryNet` model, loss function, and optimizer.  
  - Mock tests for the training loop are available in `building_blocks/tests/test_engine.py`.

### 4. **Full Pipeline Integration**  
- **File**: `building_blocks/tests/test_cytosummarynet.py`  
- **Description**: This test integrates all components (`dataset.py`, `models.py`, `engine.py`) to ensure they work together seamlessly.  
- **Getting Started**:  
  - Follow this script to see how the dataset, model, and training loop interact.  
  - Use it as a template for implementing your own end-to-end pipeline.

---

## **Pointers for Implementation**

- Start by understanding your dataset and preprocessing it to match the format expected by `TemplateDataset`.  
- Define model hyperparameters (e.g., input dimensions, layers, pooling method) in `CytoSummaryNet` to suit your data. The hyperaparameters defined in the paper are a good starting point.  
- Use the `train_loop` function to train the model, ensuring that your data, model, and loss function are compatible.  
- Explore and adapt the tests under `building_blocks/tests` to validate and debug your implementation.  

---

We hope these building blocks make it easier for you to implement CytoSummaryNet on your own dataset. If you encounter issues or have questions, please feel free to open an issue on this repository.  

# Reproducing the paper results
## To reproduce the figures shown in this paper
Inside the paper_figures folder you will find all the jupyter notebook to reproduce the figures shown in this paper. 

## To reproduce the results of this method for the cpg0004 dataset
### First install required packages:
	# If starting on an empty AWS EBS volume, otherwise ...
    sudo su
	mkdir ~/ebs_tmp/
	cd ~/ebs_tmp
	# ... start here 
	sudo yum install git -y
	sudo amazon-linux-extras install epel

### Then clone this GitHub repo:
	git clone https://github.com/broadinstitute/FeatureAggregation_single_cell.git

### Configure AWS credentials
	aws configure
_enter credentials_

### Download the LINCS metadata
	cd aws_scripts
	git init
	git remote add -f origin https://github.com/broadinstitute/lincs-cell-painting.git
	git config core.sparseCheckout true
	echo "metadata/platemaps/2016_04_01_a549_48hr_batch1/" >> .git/info/sparse-checkout
	git pull origin master

	# Download the repurposing info table to access perturbation names and MoA's (repurposing_info_long.tsv)
	cd metadata/platemaps/2016_04_01_a549_48hr_batch1
	curl -o repurposing_info_long.tsv https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/master/metadata/moa/repurposing_info_long.tsv

### Setup conda environment
	cd ..
	conda update --all
	conda create -n FAenv python=3.9 scipy=1.8 pytorch umap-learn pandas matplotlib seaborn pycytominer
	conda activate FAenv
	conda install datashader bokeh holoviews scikit-image colorcet 
	pip install kneed sklearn pytorch-metric-learning wandb tabulate

	conda env create -f environment.yml

### Download and preprocess all LINCS plates from batch 1
_edit the lincs_preprocessing_input.txt file, note that empty lines correspond to a False boolean:_
- p1: dataset name 
- p2: sqlite path 
- p3: metadata path 
- p4: barcode platemap filename 
- p5: boolean for subsample (used for developing code)
- p6: boolean for only download dose point 10 and 3.33 uM
- p7: path to aws commands text file (get_data_LINCS.txt)

_possibly edit the get_data_LINCS.txt file to download a subset of the data with "nano /aws_scripts/get_data_LINCS.txt"_
	
 	export PYTHONPATH=$PYTHONPATH:$(pwd)
	python src/Preprocess_LINCS.py @script_input_files/lincs_preprocessing_input.txt

### Train the feature aggregation model on the preprocessed plates
_modify the "script_input_files/main_LINCS_input.txt" file to the hyperparameters that you want to use:_
- p1: metadata path
- p2: wandb mode parameter
- p3: number of input features model
- p4: learning rate
- p5: epochs
- p6: number of sets per compound type
- p7: batch size (note: true batch size = p6*p7)
- p8: mean of the gaussian distribution used to sample cells (sd=800)
- p9: kFilters
- p10: minimum number of replicates for compounds to be included (default=0)


 	    export PYTHONPATH=$PYTHONPATH:$(pwd)
	    python src/main_LINCS.py @script_input_files/main_LINCS_input.txt
	
### Evaluate the trained model 
_modify the "script_input_files/fulleval_input.txt" file to the type of evaluation you want to do and on which dataset:_
- p1: number of input features model
- p2: kFilters
- p3: save newly inferred profiles as csv - boolean
- p4: evaluation mode: replicate prediction (empty) or MoA prediction (1)
- p5: dataset name
- p6: model directory
- p7: metadata path
- p8: dose point to evaluate on (10 or 3)
- p9: output directory

		export PYTHONPATH=$PYTHONPATH:$(pwd)
		python src/FullEval_CP_LINCS.py @script_input_files/fulleval_input.txt




