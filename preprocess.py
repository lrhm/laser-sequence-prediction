import scipy.io
import torch as t
import ipdb

def main():
    data = t.from_numpy(scipy.io.loadmat('Xtrain.mat')['Xtrain']).squeeze().float()
    var = t.var(data)
    mean = t.mean(data)
    norm_data = (data - mean)/var
    ipdb.set_trace()
    t.save(norm_data, 'norm_data.pt')

if __name__ == '__main__':
    main()
