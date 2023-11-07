# Used different architecture for compiling
nvcc -arch=sm_70 -o nbody 01-nbody.cu

# To run 
# ./nbody
# To profile the memory and time
# sudo nvprof ./nbody
