# Theano
Introduce theano from the beginning, so you can build your own DNN by it.
## Introduction of Theano
Theano is a Python library that lets you to define, optimize, and evaluate mathematical expressions.
Theano is especially useful for machine learning.
* [Requirements](http://deeplearning.net/software/theano/requirements.html):
	* Python
	* NumPy
	* SciPy
	* BLAS

* [How to install Theano](http://deeplearning.net/software/theano/install.html)   
## Review Machine Learning
* Define a function set (Model): f(x; w)   
	* x: input
	* w: model parameters
* Define what is the best function: Define a cost function C(f)
* Pick the best function by data: Training
	* In deep learning, this is usually done by gradient descent.
## Define function
### Overview
E.g. Define a function f(x) = x2, then compute f(-2)
```python?linenums
import theano
x = theano.tensor.scalar()
y = x**2
f = theano.function([x],y)
print f(-2)
```
Code:https://github.com/waterljwant/Theano/blob/master/Code/01_overview.py#L1-L5    
Step 0. Declare that you want to use Theano (line 1)   
Step 1. Define input variable x (line 2)   
Step 2. Define output variable y (line 3)   
Step 3. Declare the function as f (line 4)   
Step 4. Use the function f (line 5)   
We can do the same thing by python as,
```python?linenums
def f(x):
	return x**2
print f(-2)
```
So why we define a function by Theano.
It will be clear when we compute the gradients.
### Step1. Define Input Variables
```python?linenums
import theano
a = theano.tensor.scalar()
b = theano.tensor.matrix()
c = theano.tensor.matrix('ming zi')
print a
print b
print c
```
A variable can be a scalar, a matrix or a tensor   
Line 2: declare a scalar a
Line 3: declare a matrix b
Line 4: declare a matrix c with name “ming zi”

The name of a variable only make difference when you try to print the variable.
Line 7,8,9: let’s print the three variables a, b, c to see
what we get

a, b, c are symbols without any values
simplification
```python?linenums
import theano
import theano.tensor as T
a = T.scalar()
b = T.matrix()
c = T.matrix('ming zi')
print a
print b
print c
```
```dos
<TensorType(float32, scalar)>
<TensorType(float32, matrix)>
ming zi
```
### Step2. Define Output Variables
Output variables are defined based on their
relations with the input variables
• Below are some examples
```python?linenums
import theano
import theano.tensor as T
x1 = T.scalar()
x2 = T.scalar()
x3 = T.matrix()
x4 = T.matrix()
y1 = x1 + x2
y2 = x1 * x2
y3 = x3 * x4
y4 = T.dot(x3, x4)
```
y1 equals to x1 plus x2
y2 equals to x1 times x2
y3 is the elementwise multiplication of x3 and x4
y4 is the matrix multiplication of x3 and x4
### Step 3. Declare Function
```python
f = theano.function([x], y)
```
Declare the function as f
Function input: x
Function output: y
Note: the input of a function should be a list.
That is, always put the input in “[ ]”

Define the function input and output explicitly.
(equivalent to the above usage)
```python?linenums
import theano
import theano.tensor as T
x1 = T.scalar()
x2 = T.scalar()
y1 = x1 * x2
y2 = x1 **2 + x2 ** 0.5
f = theano.function([x1, x2], [y1, y2])
z = f(2,4)
print z
```
