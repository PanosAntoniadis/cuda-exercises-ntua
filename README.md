# Parallel Programming for GPUs - Matrix Multiplication

Dense Matrix Multiplication (DMM) is one of the core components in many scientific computations. In this repository, we implement the DMM algorithm for GPUs in 4 ways, increasing each time the total performance. 


## Algorithms
 - __Naive:__ Simple implementation where each thread just computes one element from the output matrix. 
 - __Coalesced memory acceses of A:__ Load tiles of the input matrix A in the shared memory.
 - __Reduced memory accesses:__ Load tiles of the input matrices A and B in the shared memory.
 - __Using cuBLAS library__


## Brief results




## Project Structure
 - __cuda:__
 - __common:__ 
 - __make:__
 - __plots:__
 - __results:__
 - __report:__

## Contributors:
- [Antoniadis Panagiotis](https://github.com/PanosAntoniadis)
- [Bazotis Nikolaos](https://github.com/Nick-Buzz)
