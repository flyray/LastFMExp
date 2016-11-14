dataset_arr = [['LastFM', './Dataset/hetrec2011-lastfm-2k/user_relation_adjacency_list.dat.part.'],
               ['Delicious', './Dataset/hetrec2011-delicious-2k/user_relation_adjacency_list.dat.part.']]
partition_arr = [['hermes1', '32', 'OMP_NUM_THREADS=32']]
# clusternum_arr = ['50', '100', '200']
# alg_arr = ['LinUCB', 'CLUB']
alpha_arr = ['0.3', '0.5', '0.7']
alpha_2_arr = ['0.05', '0.1', '0.3', '0.5', '0.7', '0.9', '1.0']
binaryRatio_arr = ['Ture', 'False']
s = [
    '#!/bin/bash',
    '#SBATCH --ntasks=1',
    '#SBATCH --cpus-per-task=',
    '#SBATCH --output=/if15/qw2ky/job_output/out_',
    '#SBATCH --error=/if15/qw2ky/job_output/error_',
    '#SBATCH --mail-type=ALL',
    '#SBATCH --mail-user=qw2ky@virginia.edu',
    '#SBATCH --nodelist=',

    'cd /if15/qw2ky/workspace/LastFMExp/',

    ' python CLUBMain.py']

partition = partition_arr[0]
for dataset in dataset_arr:
    for binaryRatio in binaryRatio_arr:
        for alpha in alpha_arr:
            for alpha_2 in alpha_2_arr:
                t = s[:]
                filename = 'CLUB_' + dataset[0] + '_' + alpha + '_' + alpha_2 + '_'
                t[2] += partition[1]
                t[3] += partition[0] + '_' + filename
                t[4] += partition[0] + '_' + filename
                t[7] += partition[0]
                t[-1] = partition[2] + t[-1] + ' --alg ' + 'CLUB' + ' --dataset ' + dataset[
                    0] + ' --alpha ' + alpha + ' --alpha_2 ' + alpha_2 + ' --binaryRatio ' + binaryRatio

                fout = open(filename + '.slurm', 'w')
                for line in t:
                    fout.write(line + '\n')
                fout.close()
