# 
# Contrastive learning for feature aggregation in image-based cell profiling
We propose a Deep Sets based method that learns the best way of aggregating single-cell feature data into a profile that better predicts a compound’s mechanism of action compared to average profiling. This is achieved by applying weakly supervised contrastive learning in a multiple instance learning setting. The proposed model provides a more accessible method for aggregating single-cell feature data than previous studies while significantly improving upon the average profiling baseline. 


All of the scripts that were used to develop, train, and evaluate the model are shown here. The scripts for computing the figures can be found in the jupyter notebooks folder.

You can find more details in the FeatureAggregationManuscriptV1.pdf



# Applying this method for your own dataset on AWS EBS
### First install required packages:
`sudo su`
`mkdir ~/ebs_tmp/`
`cd ~/ebs_tmp`
`sudo yum install git -y`
`sudo amazon-linux-extras install epel`
`sudo yum install nload sysstat parallel -y`

### Then clone this GitHub repo:
`git clone https://github.com/broadinstitute/FeatureAggregation_single_cell.git`

### Configure AWS credentials
`aws configure`
_enter credentials_

### Download all plates (in this example LINCS will be used)
`cd /aws_scripts`
`chmod +x get_data_LINCS.sh`
`./ get_data_LINCS.sh`

### Setup conda environment
cd `..`
conda env create -f environment.yml


### Preprocess all plates
python 


