import numpy as np
import scipy.sparse as sp


class ComputeSimilarity:
    def __init__(self, dataMatrix, topk=100, shrink=0, method="item", normalize=True):
        r"""Computes the cosine similarity of dataMatrix

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) :  hyper-parameter in calculate cosine distance.
            method (str) : Calculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize
        self.method = method

        self.n_rows, self.n_columns = dataMatrix.shape

        if self.method == "user":
            self.TopK = min(topk, self.n_rows)
        else:
            self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

    def compute_similarity(self, block_size=100):
        r"""Compute the similarity for the given dataset

        Args:
            block_size (int): divide matrix to :math:`n\_rows \div block\_size` to calculate cosine_distance if method is 'user',
                 otherwise, divide matrix to :math:`n\_columns \div block\_size`.

        Returns:

            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        if self.method == "user":
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
            end_local = self.n_rows
        elif self.method == "item":
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' in ['user', 'item']!")
        sumOfSquared = np.sqrt(sumOfSquared)

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:
            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if self.method == "user":
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray()

            # Compute similarities

            if self.method == "user":
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    denominator = (
                        sumOfSquared[Index] * sumOfSquared + self.shrink + 1e-6
                    )
                    this_line_weights = np.multiply(this_line_weights, 1 / denominator)

                elif self.shrink != 0:
                    this_line_weights = this_line_weights / self.shrink

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of users or items
                # - Partition the data to extract the set of relevant users or items
                # - Sort only the relevant users or items
                # - Get the original index
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[
                    0 : self.TopK
                ]
                relevant_partition_sorting = np.argsort(
                    -this_line_weights[relevant_partition]
                )
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_line_weights[top_k_idx][notZerosMask])
                if self.method == "user":
                    rows.extend(np.ones(numNotZeros) * Index)
                    cols.extend(top_k_idx[notZerosMask])
                else:
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * Index)

            start_block += block_size

        # End while
        if self.method == "user":
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_rows, self.n_rows),
                dtype=np.float32,
            )
        else:
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_columns, self.n_columns),
                dtype=np.float32,
            )
        return neigh, W_sparse.tocsc()
