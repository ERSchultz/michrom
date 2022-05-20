#! /bin/bash
#SBATCH --job-name=diff
#SBATCH --partition=depablo-ivyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=0
#SBATCH --output=logFiles/diffusion.log
#SBATCH --time=24:00:00
#SBATCH --mail-user=erschultz@uchicago.edu

dir="/project2/depablo/walt/michrom/project/chr_05/chr_05_02"
odir="/project2/depablo/erschultz/michrom/project/chr_05/chr_05_02"
jobs=10
downSampling=100
its=1
scratch='/scratch/midway2/erschultz'
chunkSize=50
plot='true'
k=2

cd /project2/depablo/erschultz/michrom
source activate python3.9_pytorch1.9_cuda10.2
module load cmake

python3 scripts/diffusion_analysis.py --dir $dir --odir $odir --jobs $jobs --down_sampling $downSampling --its $its --scratch $scratch --chunk_size $chunkSize --plot $plot --k $k
