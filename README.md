# 
# Contrastive learning for feature aggregation in image-based cell profiling (on an AWS instance)
We propose a Deep Sets based method that learns the best way of aggregating single-cell feature data into a profile that better predicts a compound’s mechanism of action compared to average profiling. This is achieved by applying weakly supervised contrastive learning in a multiple instance learning setting. The proposed model provides a more accessible method for aggregating single-cell feature data than previous studies while significantly improving upon the average profiling baseline. 


All of the scripts that were used to develop, train, and evaluate the model are shown here. The scripts for computing the figures can be found in the jupyter notebooks folder.

You can find more details in the FeatureAggregationManuscriptV1.pdf



# To apply this method yourself on the LINCS dataset
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
_edit the lincs_preprocessing_input.txt file:_
_p1: dataset name_
_p2: sqlite path_
_p3: metadata path_
_p4: barcode platemap filename_

_possibly edit the get_data_LINCS.txt file to download a subset of the data with "nano /aws_scripts/get_data_LINCS.txt"_
	
	python Preprocess_LINCS.py @script_input_files/lincs_preprocessing_input.txt

### Train the feature aggregation model on the preprocessed plates
_modify the "script_input_files/main_LINCS_input.txt" file to fit the hyperparameters you are using._

	python main_LINCS.py @script_input_files/main_LINCS_input.txt
	
### Evaluate the trained model 
_modify the "script_input_files/fulleval_input.txt" file to the type of evaluation you want to do and on which dataset._
	
	python FullEval_CP_LINCS.py @script_input_files/fulleval_input.txt




