import pandas as pd
import os
import sys
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import cv2

def main():
    """Generates folds with images according to the K-fold Cross validation, for both datasets (same and diff), with k = 5.
    """
	subjects = ['Aishwarya_Rai', 'Conan_O_Brian', 'Abdullah_II_of_Jordan', 'Cavaco_Silva',
		'Aditya_Seal', 'Anne_Princess_Royal', 'Alex_Gonzaga', 'Dalai_Lama', 'Angelique_Kidjo',
		'Zelia_Duncan', 'Aya_Miyama', 'Alain_Traore']
	orig_path = '/home/revujenation/PycharmProjects/Estagio_INESCTEC/summer_internship_dataset/dataset'

	file_list = []
	label_list = []
	for dataset in ['same', 'diff', 'test']:
		file_list[:] = []
		label_list[:] = []
		for subject in subjects:
			try:
				folder = os.path.join(os.path.join(orig_path, subject), dataset)
				for file in os.listdir(folder):
					if((file.endswith('.jpg') or file.endswith('.png')) and cv2.imread(os.path.join(folder, file)) is not None):
						shutil.copyfile(os.path.join(os.path.join(folder, file)), os.path.join(os.path.join(orig_path, dataset), '{}_{}'.format(subject, file)))
						file_list.append('{}_{}'.format(subject, file))
						label_list.append(str(subject))
			except Exception as e:
				print(e)
				pass

		if(dataset is 'test'):
			d = {'filename': file_list, 'label': label_list}
			df_test = pd.DataFrame(d)
		elif(dataset is 'diff'):
			d = {'filename': file_list, 'label': label_list}
			df_diff = pd.DataFrame(d)
		elif(dataset is 'same'):
			d = {'filename': file_list, 'label': label_list}
			df_same = pd.DataFrame(d)
		else:
			raise Exception('Not recognized dataset')

	sss = StratifiedShuffleSplit(n_splits=5, train_size=0.8, random_state=0)
	counter = 0
	# For same dataset
	for train_index, test_index in sss.split(df_same['filename'], df_same['label']):
		val_same = df_same.iloc[test_index, :]
		train_same = df_same.iloc[train_index, :]
		val_same.to_csv('val_same' + str(counter) + '.csv')
		train_same.to_csv('train_same' + str(counter) + '.csv')
		counter += 1
		# Para verificarem a racio de valores por split
		print(val_same['label'].value_counts()/len(val_same), train_same['label'].value_counts()/len(train_same))

	counter = 0
	# For different dataset
	for train_index, test_index in sss.split(df_diff['filename'], df_diff['label']):
		val_diff = df_diff.iloc[test_index, :]
		train_diff = df_diff.iloc[train_index, :]
		val_diff.to_csv('val_diff' + str(counter) + '.csv')
		train_diff.to_csv('train_diff' + str(counter) + '.csv')
		counter += 1
		# Para verificarem a racio de valores por split
		print(val_diff['label'].value_counts()/len(val_diff), train_diff['label'].value_counts()/len(train_diff))

	# Save to csv
	df_diff.to_csv('diff.csv')
	df_same.to_csv('same.csv')
	df_test.to_csv('test.csv')

if __name__ == '__main__':
	main()
