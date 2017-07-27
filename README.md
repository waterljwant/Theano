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
## Define function - Overview
E.g. Define a function f(x) = x2, then compute f(-2)
```python
import theano

x = theano.tensor.scalar()
y = x**2
f = theano.function([x],y)

print f(-2)
```
Step 0. Declare that you want to use Theano (line 1)   
Step 1. Define input variable x (line 3)   
Step 2. Define output variable y (line 4)   
Step 3. Declare the function as f (line 5)   
Step 4. Use the function f (line 7)   
