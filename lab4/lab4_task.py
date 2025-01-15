import numpy as np
class Soft_K_Means:
    def __init__(self, k, threshold=1e-6, max_iter=1000, m=2):
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter
        self.m = m

    def initialize_centroids(self, X):
        # Task 1
        # X的shape 是 178,2
        # init_centroids的shape却是 3,
        # k的值是3
        # rand_indexes 是: [93, 37, 98]---正确的
        # 问题出在np.take没有声明 axis
        indexes=np.arange(X.shape[0])
        rand_indexes=np.random.choice(indexes,self.k,replace=False)
        init_centroids=np.take(X,rand_indexes,axis=0)
        return init_centroids
        
        # tmp=rand_indexes
        # pdb.set_trace()
        # YOUR CODE HERE

    def compute_centroids(self, X, W):
        # X=(N,d) W=(N,k) find: (k,d)
        # Task 2
        k_N_d=np.tile(np.expand_dims(X,0),(self.k,1,1))
        k_N_1=np.expand_dims(W.T,-1)  #weight
        k_N_d=k_N_d * k_N_1**(self.m) #match multiply all the weights
        k_d=np.sum(k_N_d,axis=1)
        Deno=np.sum(W**(self.m),axis=0) #shape: (k,)
        centroids=k_d/Deno.reshape(-1,1)
        return centroids
        # YOUR CODE HERE
    def compute_soft_labels(self, X, C, epsilon=1e-6):
        # X=(N,d) C=(k,d) find W=(N,k)
        # Task 3
        # assert len(C.shape)==1
        # 1 build N_k distance
        tmp_X=np.expand_dims(X,1) #shape (N,1,d)
        tmp_C=np.expand_dims(C,0) #shape (1,k,d)
        dist=np.sqrt(np.sum((tmp_X-tmp_C)**2,axis=2)) #shape (N,k)
        # 2 rest part
        N_k_1=np.expand_dims(dist,-1)
        N_k_k=np.tile(np.expand_dims(dist,1),(1,self.k,1))
        N_k_k=((N_k_1+epsilon)/(N_k_k+epsilon)) **(2/((self.m)-1))
        N_k=np.sum(N_k_k,axis=2)
        weight=N_k**(-1)
        return weight
        # YOUR CODE HERE

    def fit_one_iteration(self, X, C, W):
        # assert len(C.shape)==2
        W = self.compute_soft_labels(X, C)
        C = self.compute_centroids(X, W)
        return C, W

    def fit(self, X):
        C = self.initialize_centroids(X)
        W = None
        C_prev = C
        for i in range(self.max_iter):
            C, W = self.fit_one_iteration(X, C, W)
            diff = C - C_prev
            if np.mean(np.sqrt(np.sum(diff ** 2, axis=1))) <= self.threshold:
                break
            C_prev = C
        else:
            print(f'[WARNING] max iter {self.max_iter} reached')
        return C, W