# causal-optimization

## linear causal discovery as a graph matching problem 

### causal discovery using continuous optimization methods:

The optimization method is based on sequential convex quadratic programming. This method benefits from searching in the space of permutation matrices, which is considerably smaller than the search space of DAGs. It is also based on a two-step optimization procedure, in which the convexity of one step was proved and the closed-form solution of the other step was found.

- Algorithm:

![seq](https://github.com/aliizadi/causal-optimization/blob/main/imgs/1.png)

- Result:

![seq](https://github.com/aliizadi/causal-optimization/blob/main/imgs/2.png)
