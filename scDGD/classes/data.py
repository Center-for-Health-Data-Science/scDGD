import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse

class scDataset(Dataset):
    """
    """
    def __init__(self, sparse_mtrx, meta_data, scaling_type='mean', gene_selection=None, subset=None, label_type='stop', sparse=False):
        """
        Args:
        This is a custom data set for single cell transcriptomics data.
        It takes a sparse matrix of gene expression data and a pandas dataframe of metadata.

        sparse_mtrx: 
            a scipy.sparse matrix of gene expression data with rows representing cells and columns representing transcripts
        meta_data: 
            a pandas dataframe of metadata with rows representing cells and columns representing metadata
        scaling_type: 
            a string specifying the type of scaling to use for the data. Options are 'mean' and 'max'
            this will either scale the data by the mean or max of each cell
        gene_selection (optional): 
            a list of indices specifying which genes to use from the sparse matrix if feature selection is to be performed
        subset (optional): 
            a list of indices specifying which cells to use from the sparse matrix if subsampling is to be performed
        label_type (optional):
            the label type of the characteristic that one wants to observe in clustering. It is usually used for cell type.
            It indicates the column name of the meta data provided.
        sparse (optional):
            a boolean indicating whether the data should be kept in sparse format or converted to a dense tensor.
            For small data sets, it is recommended to transform the data in dense format for faster training.
            For large ones, this helps keep the memory used in check.
        """

        self.scaling_type = scaling_type
        self.meta = meta_data

        if gene_selection is not None:
            sparse_mtrx = sparse_mtrx.tocsc()[:,gene_selection].tocoo()
        if subset is not None:
            sparse_mtrx = sparse_mtrx.tocsr()[subset]
        if sparse:
            self.sparse = True
            sparse_mtrx = sparse_mtrx.tocsr()
            self.data = sparse_mtrx
            if self.scaling_type == 'mean':
                self.library = torch.tensor(sparse_mtrx.mean(axis=-1).toarray())
            elif self.scaling_type == 'max':
                self.library = torch.tensor(sparse_mtrx.max(axis=-1).toarray())

        else:
            self.sparse = False
            self.data = torch.Tensor(sparse_mtrx.todense())
            if self.scaling_type == 'mean':
                self.library = torch.mean(self.data, dim=-1).unsqueeze(1)
            elif self.scaling_type == 'max':
                self.library = torch.max(self.data, dim=-1).values.unsqueeze(1)
        
        self.n_genes = self.data.shape[1]
        
        self.label_type = label_type

    def __len__(self):
        return(self.data.shape[0])

    def __getitem__(self, idx=None):
        if idx is not None:
            if self.sparse:
                expression = self.data[idx]
            else:
                expression = self.data[idx]
        else:
            expression = self.data
            idx = torch.arange(self.data.shape[0])
        lib = self.library[idx]
        return expression, lib, idx
    
    def get_labels(self, idx=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx is None:
            idx = np.arange(self.__len__())
        if self.label_type == 'stop':
            label_ids = np.argmax(np.expand_dims(np.asarray(self.meta['stop']),0)>=np.expand_dims(idx,1),axis=1)
            return np.asarray(np.array(self.meta['label'])[label_ids])
        elif self.label_type == 'cell_type':
            return np.asarray(np.array(self.meta['cell_type'])[idx])
    
    def get_labels_numerical(self, idx=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx is None:
            idx = np.arange(self.__len__())
        label_ids = np.argmax(np.expand_dims(np.asarray(self.meta['stop']),0)>=np.expand_dims(idx,1),axis=1)
        return label_ids
    

###
# functions used for the sparse option
# here the data will have to be transformed to dense format when a batch is called
###

def sparse_coo_to_tensor(mtrx):
    return torch.FloatTensor(mtrx.todense())

def collate_sparse_batches(batch):
    data_batch, library_batch, idx_batch = zip(*batch)
    data_batch = scipy.sparse.vstack(list(data_batch))
    data_batch = sparse_coo_to_tensor(data_batch)
    library_batch = torch.stack(list(library_batch), dim=0)
    idx_batch = list(idx_batch)
    return data_batch, library_batch, idx_batch
