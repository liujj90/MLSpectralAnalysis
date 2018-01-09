## load libraries
import math
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model
import os

def avg_over_col(data, num_col):
	new_df = pd.DataFrame()
	curr_col = 0
	for i in range(num_col,len(data.columns)+1, num_col):
		working_df = data.iloc[:,curr_col:i]
		colname = str(working_df.columns[0])
		new_df[colname]=working_df.mean(axis = 1)
		curr_col += num_col
	return(new_df)


class run_stored_model(object):
	def __init__(self, filename, *args, **kwargs):
		self.dir = os.getcwd()
		self.norms = joblib.load(self.dir+ '/normalizer.save')
		self.stds = joblib.load(self.dir+'/standarizer.save')
		self.filename = self.dir+filename
		self.sigmoid_model_n = load_model(self.dir+'/sigmoid_norm.h5')
		self.sigmoid_L1_model_n =load_model(self.dir+'/sigmoid_L1_norm.h5')
		self.sigmoid_model_s = load_model(self.dir+'/sigmoid_std.h5')
		self.sigmoid_L1_model_s = load_model(self.dir+'/sigmoid_L1_std.h5')
		self.df = self.read_pre_process()

    # read file, output values averaged over 4 wavelengths, wide format
	def read_pre_process(self): 
	    file = pd.read_table(self.filename, header = None, index_col = None )
	    file.columns = ['to_del','Wavelength', 'Intensity']
	    file = file.drop('to_del', axis = 1)
	    file['Wavelength']=file['Wavelength'].apply(lambda x: math.floor(x))
	    file = file.groupby(['Wavelength']).agg(np.mean).reset_index()
	    file = file[file['Wavelength'].between(450, 850)].set_index('Wavelength')
	    df = file.transpose()
	    return df
	# predict model
	def preprocess_predict(self):
		df_norm_1 =  pd.DataFrame(self.norms.transform(self.df.values), columns = self.df.columns)
		df_std_1 = pd.DataFrame(self.stds.transform(self.df.values), columns = self.df.columns)
		df_norm = avg_over_col(data = df_norm_1, num_col = 4)
		df_std = avg_over_col(data = df_std_1, num_col = 4)
		sign = [x for y in self.sigmoid_model_n.predict(df_norm).tolist() for x in y]
		sigs = [x for y in self.sigmoid_L1_model_n.predict(df_norm).tolist() for x in y]
		sign_L = [x for y in self.sigmoid_model_s.predict(df_std).tolist() for x in y]
		sigs_L = [x for y in self.sigmoid_L1_model_s.predict(df_std).tolist() for x in y]
		return (sign+sigs+sign_L+sigs_L)

print(run_stored_model('/test.txt').preprocess_predict())