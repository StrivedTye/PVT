import torch
from pointnet2.utils import pointnet2_utils


@torch.no_grad()
def eig_vector(data, K):
    b, n, _ = data.shape
    idx = pointnet2_utils.knn_point(K, data, data)  # idx (b,n,K)

    idx0 = torch.arange(0, b, device=data.device).reshape((b, 1)).expand(-1, n * K).reshape((1, b * n * K))
    idx1 = torch.arange(0, n, device=data.device).reshape((1, n, 1)).expand(b, n, K).reshape((1, b * n * K))
    idx = idx.reshape((1, b * n * K))
    idx = torch.cat([idx0, idx1, idx], dim=0)  # (3, b*n*K)
    # print(b, n, K, idx.shape)
    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense()  # (b,n,n)
    A = A | A.transpose(1, 2)
    A = A.float()
    deg = torch.diag_embed(torch.sum(A, dim=2))
    laplacian = deg - A
    u, v = torch.linalg.eig(laplacian)  # (b,n,n)
    return v.real, laplacian, u.real
