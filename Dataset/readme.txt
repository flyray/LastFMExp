tfidf_generator_xxx.py generates the 25-dim PCA result of tf-idf matrix gathered from user_taggedxxx-timestamps.dat.

To run the script, put it in the same folder with hetrec2011-xxx-2k/, and run command:
python tfidf_generator_xxx.py hetrec2011-xxx-2k/

There are two file output, arm_idx.dat & feature_vectors.dat.

arm_idx.dat:
	Projection between row index in feature matrix and arm id.

feature_vectors.dat:
	Feature matrix which reduced column dimension to 25.
	Use scipy.io.mmread() to read the file.