Intro to Linear Algebra with Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 R and R Studio](#01-r-and-r-studio)
	+ [0.2 Packages](#02-packages)
- [1.0 Background](#10-background)
	+ [1.1 Machine Learning](#11-Machine Learning)
	+ [1.2 Data](#12-data)
	+ [1.3 Overfitting vs Underfitting](#13-overfitting-vs-underfitting)
	+ [1.4 Glossary](#14-glossary)
		* [1.4.1 Factors](#141-factors)
		* [1.4.2 Corpus](#142-corpus)
		* [1.4.3 Bias](#143-bias)
		* [1.4.4 Variance](#144-variance)
- [2.0 Data Preparation](#30-data-preparation)
	+ [2.1 dplyr](#31-dplyr)
	+ [2.2 Geopandas](#32-geopandas)
- [3.0 Exploratory Analysis](#30-exploratory-analysis)
- [4.0 Data Visualization](#50-data-visualization)
- [5.0 Machine Learning & Prediction](#50-machine-learning--prediction)
	+ [5.1 Random Forests](#51-random-forests)
	+ [5.2 Natural Language Processing](#52-natural-language-processing)
		* [5.2.1 ANLP](#521-anlp)
	+ [5.3 K Means Clustering](#53-k-means-clustering)
- [6.0 Final Exercise]($60-final-exercise)
- [7.0 Final Words](#60-final-words)
	+ [7.1 Resources](#61-resources)
	+ [7.2 More!](#72-more)

## 0.0 Setup

This guide was written in Python 3.5.


### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).


### 0.2 Packages


```
pip3 install scipy
pip3 install numpy
```

## 1.0 Introduction

Linear Algebra is a branch of mathematics that lets you concisely describe coordinates and interactions of planes in higher dimensions and perform operations on them. 

Think of it as an extension of algebra into an arbitrary number of dimensions. Linear Algebra is about working on linear systems of equations (linear regression is an example: y = Ax). Rather than working with scalars, we start working with matrices and vectors (vectors are really just a special type of matrix). Vectors are the core of linear algebra studies. 


## 2.0 Vectors

### 2.1 What is a vector? 

Simply put, a vector is an ordered tuple of numbers which have both a magnitude and direction. It's important to note that vectors are an <b>element</b> of a vector space. 

In section 3, we'll learn about matrices, which are a rectangular array of values. A vector is simply a one dimensional matrix. 

In Python, we can represent a vector with a list of lists, for example:  

``` python
A = [[1.0,2.0],[3.0,4.0]]
```

In many instances, matrices can be made with numpy arrays.

``` python
import numpy as np
A = np.array([[1.0,2.0],[3.0,4.0]])
```

#### 2.2 What is a vector space?

A vector space, &Nu; is a set that contains all linear combinations of its elements. Therefore:

- If vectors u and v &isin; &Nu;, then u + v &Nu;  
- If u &isin; &Nu;, then &alpha; u &isin; &Nu; for any scalar &alpha;
- There exists 0 &isin; &Nu; such that u + 0 = u for any u &isin; &Nu;

### 2.3 What is a subspace?

A subspace is a subset of a vector space that is also a vector space. 

### 2.4 What is a linear independence? 

A vector u is linear independent of a set of vectors if it does <b>not</b> lie in their <b>span</b>. A set of vectors is linearly independent if every vector is linearly independent of the rest. 

### 2.5 What is a basis? 

A basis of a vector space &Nu; is a linearly independent set of vectors whose span is <b>equal</b> to &Nu;. If a vector space has a basis with d vectors, its dimension is d. 

### 2.6 What is a Norm? 

A norm just refers to the magnitude of a vector, and is denoted with ||u||. With numpy and scipy, we can do calculate the norm as follows: 
``` python
import numpy as np
from scipy import linalg
# norm of a vector
v = np.array([1,2])
linalg.norm(v)
```

## 3.0 Matrices

A Matrix is a 2D array that stores real or complex numbers. You can use numpy to create matrices:

``` python
import numpy as np
matrix1 = np.matrix(
    [[0, 4],
     [2, 0]]
)
matrix2 = np.matrix(
    [[-1, 2],
     [1, -2]]
)
```

### 3.1 Identity Matrix

A Diagonal Matrix is an n x n matrix with 1s on the diagonal from the top left to the bottom right, such as 

``` 
[[ 1., 0., 0.],
[ 0., 1., 0.],
[ 0., 0., 1.]]

```
We can generate diagonal matrices with the `eye()` function in Python: 

``` python
np.eye(4)
```

When a matrix is multiplied by its inverse, the result is the identity matrix. It's important to note that only square matrices have inverses.

### 3.2 Matrix Operations

#### 3.2.1 Addition

Matrix addition works very similarlty to normal addition. You simply add the corresponding spots together. 
``` python
matrix_sum = matrix1 + matrix2
```
And you'll get
``` 
matrix([[-1,  6],
        [ 3, -2]])
```

#### 3.2.2 Multiplication


``` python
matrix_prod = matrix1 * matrix2
```

#### 3.2.3 Determinant

``` python
det = np.linalg.det(matrix1)
```

#### 3.2.3 Inverse

``` python
inverse = np.linalg.inv(matrix1)
```

#### 3.2.4 Eigenvalues

``` python
eigvals = np.linalg.eigvals(matrix)
```

#### 3.2.5 Solving Systems of Equations

Consider a set of m linear equations in n unknowns:

We can re-write the system:
```
Ax = b
```
This reduces the problem to a matrix equation, and now solving the system amounts to finding A<sub>−1</sub>.

### 3.3 Underdetermined Matrices

When `m<n`, the linear system is said to be <b>underdetermined</b>, e.g. there are fewer equations than unknowns. In this case, there are either no solutions or infinite solutions and a unique solution is not possible.

### 3.5 Kernels

The kernel of a matrix A is the dimension of the space mapped to zero under the linear transformation that A represents. The dimension of the kernel of a linear transformation is called the nullity.

## 5.0 Final Words


### 5.1 Resources

[]() <br>
[]()
