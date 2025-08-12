import numpy as np
import scipy.sparse as sp
import json
import os
class FisherData:
    def __init__(self, row_dim, col_dim, nnz, w, u, b, x0, row_ptr, col_ind, power = 1.0):
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.nnz = nnz
        self.power = power
        self.w = w
        self.u = u
        self.b = b
        self.x0 = x0
        self.row_ptr = row_ptr
        self.col_ind = col_ind


    
def load_fisher_ces_data(meta_file_dir):
    meta_file = os.path.join(meta_file_dir, 'fisher_ces_meta.json')
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    b = np.load(os.path.join(meta_file_dir, meta['b']))
    col_ind = np.load(os.path.join(meta_file_dir, meta['col_ind']))
    row_ptr = np.load(os.path.join(meta_file_dir, meta['row_ptr']))
    u = np.load(os.path.join(meta_file_dir, meta['u_val']))
    w = np.load(os.path.join(meta_file_dir, meta['w']))
    x0 = np.load(os.path.join(meta_file_dir, meta['x0']))
    power = meta['power']
    row_dim = meta['row_dim']
    col_dim = meta['col_dim']
    nnz = meta['nnz']
    return FisherData(row_dim, col_dim, nnz, w, u, b, x0, row_ptr, col_ind, power)
    
    
    return 

if __name__ == "__main__":
    # 示例用法
    meta_file_dir = 'file_dir/problem/ces_row_1000_col_400_nnz_80000_0.500000/20250804_100541_022'
    fisher_data = load_fisher_ces_data(meta_file_dir)
    