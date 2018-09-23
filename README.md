# AdaGram-Cpp

Adaptive Skip-gram (AdaGram) model is a nonparametric extension of famous Skip-gram model implemented in word2vec software
which is able to learn multiple representations per word capturing different word meanings. 
This is C++ ported version of AdaGram in Julia (https://github.com/sbos/AdaGram.jl).

You can see more informations about AdaGram at this repository(https://github.com/sbos/AdaGram.jl).

## Requirements

* Eigen (C++ Library) : http://eigen.tuxfamily.org

## Performance

By using Eigen, This code improves performance up to ~x2 compared to original Julia version,
~x3 when compiling with SSE2, and ~x5 when compiling with AVX.
