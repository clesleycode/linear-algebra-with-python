Intro to Linear Algebra with Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python and Pip](#01-python-and-pip)
	+ [0.2 Libraries](#02-libraries)
- [1.0 Introduction](#10-introduction)
- [2.0 Vectors](#30-vectors)
	+ [2.1 What is a vector?](#21-what-is-a-vector)
	+ [2.2 What is a vector space?](#22-what-is-a-vector-space)
	+ [2.3 What is a subspace?](#23-what-is-a-subspace)
	+ [2.4 What is linear independence?](#24-what-is-linear-independence)
	+ [2.5 What is a basis?](#25-what-is-a-basis)
	+ [2.6 What is a Norm?](#26-what-is-a-norm)
- [3.0 Matrices](#30-matrices)
	+ [3.1 Identity Matrix](#31-identity-matrix)
	+ [3.2 Matrix Operations](#32-matrix-operations)
		* [3.2.1 Addition](#321-addition)
		* [3.2.2 Multiplication](#322-multiplication)
		* [3.2.3 Determinant](#323-determinant)
		* [3.2.4 Inverse](#324-inverse)
		* [3.2.5 Eigenvalues](#325-eigenvalues)
		* [3.2.6 Solving Systems of Equations](#326-solving-systems-of-equations)
	+ [3.3 Underdetermined Matrices](#33-underdetermines-matrices)
	+ [3.5 Kernels](#35-kernels)
- [7.0 Final Words](#60-final-words)
	+ [7.1 Resources](#61-resources)
	+ [7.2 More!](#72-more)

## 0.0 Setup

This guide was written in Python 3.5.


### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).


### 0.2 Libraries


```
pip3 install scipy
pip3 install numpy
```

## 1.0 Introduction

Linear Algebra is a branch of mathematics that lets you concisely describe coordinates and interactions of planes in higher dimensions and perform operations on them. 

Think of it as an extension of algebra into an arbitrary number of dimensions. Linear Algebra is about working on linear systems of equations. Rather than working with scalars, we start working with matrices and vectors. Vectors are the core of linear algebra studies. 

### 1.1 Why Learn Linear Algebra?

<b>Machine Learning</b>: A lot of Machine Learning concepts are tied to linear algebra concepts. Some basic examples, PCA - eigenvalue, regression - matrix multiplication. As most ML techniques deal with high dimensional data, they are often times represented as matrices.

<b>Mathematical Modeling</b>: for example, if you want to capture behaviors (sales, engagement, etc.) in a mathematical model, you can use matrices to breakdown the samples into their own subgroups. This requires some basic matrix manipulation, such as atrix inversion, derivation, solving partial differential, or first order differential equations with matrices, for example. 


## 2.0 Vectors

### 2.1 What is a vector? 

Simply put, a vector is an ordered tuple of numbers which have both a magnitude and direction. It's important to note that vectors are an <b>element</b> of a vector space. 

In section 3, we'll learn about matrices, which are a rectangular array of values. A vector is simply a one dimensional matrix. 

In Python, we can represent a vector with a list of lists, for example:  

``` python
A = [1.0,2.0,3.0,4.0]
```

In many instances, matrices can be made with numpy arrays.

``` python
import numpy as np
A = np.array([1.0,2.0,3.0,4.0])
```

#### 2.2 What is a vector space?

A vector space, &Nu; is a set that contains all linear combinations of its elements. Therefore:

- If vectors u and v &isin; &Nu;, then u + v &Nu;  
- If u &isin; &Nu;, then &alpha; u &isin; &Nu; for any scalar &alpha;
- There exists 0 &isin; &Nu; such that u + 0 = u for any u &isin; &Nu;

### 2.3 What is a subspace?

A subspace is a subset of a vector space that is also a vector space. 

### 2.4 What is linear independence? 

A collection of vectors v<sub>1</sub>,...,v<sub>n</sub> is said to be linearly independent if:

c<sub>1</sub>v<sub>1</sub> + ... + c<sub>n</sub>v<sub>n</sub> = 0

c<sub>1</sub> = ... = c<sub>n</sub> = 0

In other words, any linear combination of the vectors that results in a zero vector is trivial.

### 2.5 What is a basis? 

A basis of a vector space is any linearly independent subset of it that spans the whole vector space.  In other words, each vector in the vector space can be written exactly in one way as a linear combination of the basis vectors.

### 2.6 What is a Norm? 

A norm just refers to the magnitude of a vector, and is denoted with ||u||. With numpy and scipy, we can do calculate the norm as follows: 

``` python
import numpy as np
from scipy import linalg
# norm of a vector
v = np.array([1,2])
linalg.norm(v)
```
The actual formula looks like: 

![alt text](https://github.com/lesley2958/lin-alg/blob/master/norm.png?raw=true "Logo Title Text 1")

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

Visually, this looks like:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/vector%20addition.png?raw=true "Logo Title Text 1")

#### 3.2.2 Multiplication

The dot product is an algebraic operation that takes two coordinate vectors of equal size and returns a single number. The result is calculated by multiplying corresponding entries and adding up those products. 

To multiply two matrices, you will use this dot product pattern. With numpy, you can use the `np.dot` method: 

``` python
np.dot(matrix2, matrix1)
```
Or, simply, you can do:

``` python
matrix_prod = matrix1 * matrix2
```

#### 3.2.3 Trace and Determinant

The trace of a matrix A is the sum of its diagonal elements. It's important because it's an invariant of a matrix under change of basis and it defines a matrix norm. 

The determinant of a matrix is defined to be the alternating sum of permutations of the elements of a matrix. The formula is as follows:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/det.png?raw=true "Logo Title Text 1")

In python, you can use the following function: 

``` python
det = np.linalg.det(matrix1)
```
Note that an n×n matrix Ais invertible &iff; det(A) &ne; 0.

#### 3.2.4 Inverse

The matrix A is invertible if there exists a matrix A<sub>-1</sub> such that

A<sub>-1</sub>A = I and AA<sub>-1</sub> = I

Multiplying inverse matrix is like the division; it’s simply the reciprocal function of multiplication. Let’s say here is a square matrix A, then multiplying its inversion gives the identity matrix I.

We can get the inverse matrix with numpy: 

``` python
inverse = np.linalg.inv(matrix1)
```

#### 3.2.5 Eigenvalues & Eigenvectors

Let A be an n x n matrix. The number &lambda; is an eigenvalue of A if there exists a non-zero vector C such that

Av = &lambda;v

In this case, vector v is called an eigenvector of A corresponding to &lamda;. You can use numpy to calculate the eigenvectors of a matrix: 

``` python
eigvals = np.linalg.eigvals(matrix)
```

Note that eigenvectors do not change direction in the transformation of the matrix.

#### 3.2.6 Solving Systems of Equations

Consider a set of m linear equations in n unknowns:

We can re-write the system:
```
Ax = b
```
This reduces the problem to a matrix equation, and now solving the system amounts to finding A<sub>−1</sub>.

### 3.3 Under vs Overdetermined Matrices

When `m < n`, the linear system is said to be <b>underdetermined</b>, e.g. there are fewer equations than unknowns. In this case, there are either no solutions or infinite solutions and a unique solution is not possible.

When `m > n`, the system may be <b>overdetermined</b>. In other words, there are more equations than unknowns. They system could be inconsistent, or some of the equations could be redundant. 

### 3.4 Row, Column, and Null Space 

The <b>column space</b> C(A) of a matrix A (sometimes called the range of a matrix) is the span (set of all possible linear combinations) of its column vectors.

The <b>row space</b> of an m x n matrix, A, denoted by R(A) is the set of all
linear combinations of the row vectors of A.

The <b>null space</b> of an m x n matrix, A, denoted by null(A) is the set of all solutions, x, of the equation Ax = 0<sub>m</sub>.

### 3.5 Rank

The rank of a matrix A is the dimension of its column space - and - the dimension of its row space. These are equal for any matrix. Rank can be thought of as a measure of non-degeneracy of a system of linear equations, in that it is the dimension of the image of the linear transformation determined by A.

### 3.6 Kernels

The kernel of a matrix A is the dimension of the space mapped to zero under the linear transformation that A represents. The dimension of the kernel of a linear transformation is called the nullity.

### 3.7 Matrix Norms

We can extend the notion of a norm of a vector to a norm of a matrix. Matrix norms are used in determining the condition of a matrix. There are many matrix norms, but three of the most common are so called ‘p’ norms, and they are based on p-norms of vectors.

So, for an n-dimensional vector v and for 1 &le; p &le; &infin;, we have the following formula:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/p%20norm.png?raw=true "Logo Title Text 1")

And for p = &infin;:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/infty%20norm.png?raw=true "Logo Title Text 1")

The corresponding matrix norms are:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/matrix%20norms.png?raw=true "Logo Title Text 1")
