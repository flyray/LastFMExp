dataset_arr = [['LastFM', './Dataset/hetrec2011-lastfm-2k/user_relation_adjacency_list.dat.part.'],['Delicious', './Dataset/hetrec2011-delicious-2k/user_relation_adjacency_list.dat.part.']]
partition_arr = [['parallel','16','48:00:00','OMP_NUM_THREADS=16'], ['serial','1', '120:00:00', 'OMP_NUM_THREADS=8']]
clusternum_arr = ['50', '100', '200']
alg_arr = ['LinUCB', 'M_LinUCB', 'Uniform_LinUCB', 'CoLinUCB', 'GOBLin']
diagnol_arr = ['Opt', 'Max', '0.3' 'Org']

s = [
'#!/bin/bash',
'#SBATCH --ntasks=1',
'#SBATCH --cpus-per-task=',
'#SBATCH --time=',
'#SBATCH -p ',
'#SBATCH --output=/nv/blue/hw7ww/job_output/out_',
'#SBATCH --error=/nv/blue/hw7ww/job_output/error_',
'#SBATCH --mail-type=ALL',
'#SBATCH --mail-user=hw7ww@virginia.edu',

'cd /scratch/hw7ww/Bandit/Code/LastFMExp',

' python ./Main.py']
 
diagnol = diagnol_arr[0]
partition = partition_arr[0]
for dataset in dataset_arr:
	for num in clusternum_arr:
		for alg in alg_arr[-2:]:
			t = s[:]
			filename = partition[0]+'_'+dataset[0]+'_'+num+'_'+alg+'_'+diagnol
			t[2] += partition[1]
			t[3] += partition[2]
			t[4] += partition[0]
			t[5] += partition[0]+'_'+filename
			t[6] += partition[0]+'_'+filename
			t[-1] = partition[3]+t[-1]+' --alg '+alg+' --dataset '+dataset[0]+' --clusterfile '+dataset[1]+num+' --diagnol '+diagnol

			fout = open(filename+'.slurm', 'w')
			for line in t:
				fout.write(line+'\n')
			fout.close()

partition = partition_arr[1]
for dataset in dataset_arr:
	for num in clusternum_arr:
		for alg in alg_arr[:-2]:
			t = s[:]
			filename = partition[0]+'_'+dataset[0]+'_'+num+'_'+alg+'_'+diagnol
			t[2] += partition[1]
			t[3] += partition[2]
			t[4] += partition[0]
			t[5] += partition[0]+'_'+filename
			t[6] += partition[0]+'_'+filename
			t[-1] = partition[3]+t[-1]+' --alg '+alg+' --dataset '+dataset[0]+' --clusterfile '+dataset[1]+num+' --diagnol '+diagnol

			fout = open(filename+'.slurm', 'w')
			for line in t:
				fout.write(line+'\n')
			fout.close()		