#! /bin/bash
#SBATCH --job-name=corr
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=0
#SBATCH --output=logFiles/correlation.log
#SBATCH --time=24:00:00
#SBATCH --mail-user=erschultz@uchicago.edu

dir="/project2/depablo/walt/michrom/project/chr_05/chr_05_02"
odir="/project2/depablo/erschultz/michrom/project/chr_05/chr_05_02"
dir="/home/erschultz/dataset_test/samples/sample1"
odir="/home/erschultz/dataset_test/samples/sample1"
jobs=20
jobs=10
downSampling=12
downSampling=10
updateMode='kNN'
k=4

cd /project2/depablo/erschultz/michrom
source activate python3.9_pytorch1.9_cuda10.2
module load cmake

python3 scripts/correlation_analysis.py --dir $dir --odir $odir --jobs $jobs --down_sampling $downSampling --update_mode $updateMode --k $k
