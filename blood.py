import numpy, pandas, sys, os, scipy.stats, sklearn.decomposition, sklearn.cluster
from sklearn.mixture import GaussianMixture

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
import plotly.figure_factory as ff

def transform_to_percentile(dataframe):

    '''
    Apparently this is properly called the spearman rank
    '''

    transformed_dataframe = (dataframe.shape[0] - dataframe.rank(axis=0, ascending=False, na_option='bottom')+1)/dataframe.shape[0]

    return transformed_dataframe

# Read in data and metadata - there are two metadata files at the moment because of general disorganisation

all_data        = pandas.read_hdf('/path/to/data', 'all_data')
s4m_metadata    = pandas.read_hdf('/path/to/metadata', 'metadata')
all_annotations = pandas.read_csv('/path/to/data_portal_metadata/blood_atlas_annotations_130519.tsv', sep='\t').set_index('sample_id') # this is from data portal

# Reorganising slightly disorganised annotations - Data Portal does not yet have platform, the s4m metadata has a column 'Platform_Category' which is used as the platform variable

all_annotations       = all_annotations.loc[(all_annotations['include_blood'].values==True)]

all_annotations.index = numpy.core.defchararray.add(all_annotations.index.values.astype(str), numpy.full(shape=all_annotations.shape[0], fill_value=';').astype(str))
all_annotations.index = numpy.core.defchararray.add(all_annotations.index.values.astype(str), all_annotations['dataset_id'].values.astype(str))

blood_annotations = all_s4m_metadata.merge(all_annotations[['display_metadata', 'tier1', 'tier2', 'tier3', 'celltype']], how='right', left_index=True, right_index=True)

# Select only the blood data, and drop genes that are not measured in all datasets
blood_data       = all_data.iloc[:,numpy.in1d(all_data.columns.values.astype(str), blood_annotations.index.values.astype(str))].dropna(how='any')

# Search for platform dependent genes 

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri, rpy2.robjects.pandas2rii
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

variancePartition = importr('variancePartition')
stats             = importr('stats')

form              = robjects.Formula('~ Platform_Category')
varPart           = variancePartition.fitExtractVarPartModel(transform_to_percentile(blood_data), form, blood_annotations[['Platform_Category']])

sel_varPart   = numpy.array(varPart)[0] <= 0.2 # This is the thresholding, at a value of 0.2

genes_to_keep = blood_data.index.values[sel_varPart]

cut_data      = transform_to_percentile(blood_data.loc[genes_to_keep].copy())

# Perform PCA 

pca        = sklearn.decomposition.PCA(n_components=10, svd_solver='full')
output = pca.fit_transform(cut_data.transpose())

# Example reading in new data

new_data = pandas.read_csv('/path/to/new_data')
new_data = transform_to_percentile(new_data.loc[genes_to_keep])

new_output = pca.transform(new_data.transpose())
