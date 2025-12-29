[![CMake build](https://github.com/ConSol-Lab/contree/actions/workflows/cmake.yml/badge.svg)](https://github.com/ConSol-Lab/contree/actions/workflows/cmake.yml)

# ConTree: Optimal Classification Trees for Continuous Feature Data
Cătălin E. Briţa, Jacobus G. M. van der Linden [(e-mail)](mailto:J.G.M.vanderLinden@tudelft.nl), Emir Demirović -
Delft University of Technology

ConTree computes optimal binary classification trees on datasets with continuous features using dynamic programming with branch-and-bound.

If you use ConTree, please cite our paper:
* Briţa, Cătălin E., Jacobus G. M. van der Linden, and Emir Demirović. "Optimal Classification Trees for Continuous Feature Data Using Dynamic Programming with Branch-and-Bound." In _Proceedings of AAAI-25_ (2025). [pdf](https://arxiv.org/pdf/2501.07903)

## Usage

### Compiling
The code can be compiled on Windows or Linux by using cmake. For Windows users, cmake support can be installed as an extension of Visual Studio and then this repository can be imported as a CMake project.

For Linux users, they can use the following commands:

```sh
cd code
mkdir build
cd build
cmake ..
cmake --build .
```
The compiler must support the C++17 standard

### Running
After ConTree is built, the following command can be used (for example):
```sh
./ConTree -file ../datasets/bank.txt -max-depth 3
```

Run the program without any parameters to see a full list of the available parameters.

## Parameters
ConTree can be configured by the following parameters:
* `max_depth` : The maximum depth of the tree. Note that a tree of depth zero has a single leaf node. A tree of depth one has one branching node and two leaf nodes.
* `max_gap` : The maximum permissible gap to the optimal solution.
* `max_gap_decay` : Use this parameter, if you want to find solutions iteratively, with each iteration decreasing the `max_gap` by multiplying it with `max_gap_decay`.
* `time_limit` : The run time limit in seconds. If the time limit is exceeded a possibly non-optimal tree is returned.
* `sort_gini` : If true, the features are sorted by gini impurity.
* `use_upper_bound` : Enables or disables the use of upper bounds.
* `verbose` : Enable or disable verbose output.

## Miscellaneous
ConTree assumes classification labels are in the range `0 ... n_labels - 1`. Not meeting this assumption may influence the algorithm's performance. Ensure your input data has properly encoded labels in this range.


## Related Work
This work is follow up on our previous research:
* Demirović, Emir, et al. "Murtree: Optimal decision trees via dynamic programming and search." _Journal of Machine Learning Research_ 23.26 (2022): 1-47. [pdf](https://www.jmlr.org/papers/volume23/20-520/20-520.pdf) / [source](https://bitbucket.org/EmirD/murtree/src/master/)
* Van der Linden, Jacobus G. M., Mathijs M. de Weerdt, and Emir Demirović. "Necessary and Sufficient Conditions for Optimal Decision Trees using Dynamic Programming." In _Advances in Neural Information Processing Systems_ (2023). [pdf](https://arxiv.org/pdf/2305.19706) / [source](https://github.com/AlgTUDelft/pystreed)

Other related work:
* Hu, Xiyang, Cynthia Rudin, and Margo Seltzer. "Optimal sparse decision trees." In _Advances in Neural Information Processing Systems_ (2019). [pdf](https://proceedings.neurips.cc/paper_files/paper/2019/file/ac52c626afc10d4075708ac4c778ddfc-Paper.pdf) / [source](https://github.com/xiyanghu/OSDT)
* Lin, Jimmy, et al. "Generalized and scalable optimal sparse decision trees." In _International Conference on Machine Learning_ (2020). [pdf](https://proceedings.mlr.press/v119/lin20g/lin20g.pdf) / [source](https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees)
* Aglin, Gaël, Siegfried Nijssen, and Pierre Schaus. "Learning optimal decision trees using caching branch-and-bound search." In _Proceedings of the AAAI conference on artificial intelligence_ (2020). [pdf](https://ojs.aaai.org/index.php/AAAI/article/download/5711/5567) / [source](https://github.com/aia-uclouvain/pydl8.5)
* Mazumder, Rahul, Xiang Meng, and Haoyue Wang. "Quant-BnB: A scalable branch-and-bound method for optimal decision trees with continuous features." In _International Conference on Machine Learning_ (2022). [pdf](https://proceedings.mlr.press/v162/mazumder22a/mazumder22a.pdf) / [source](https://github.com/mengxianglgal/Quant-BnB)
