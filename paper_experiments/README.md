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
	git clone https://github.com/broadinstitute/2024_vanDijk_PLoS_CytoSummaryNet.git

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




