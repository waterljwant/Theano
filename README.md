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
### Step 4. Use Function
Line 12: simply use the function f you declared as a normal
python function
Line 13: Print the function output ->

(The theano function output is a numpy.ndarray.)
Examples for Matrix
Be careful that the dimensions of the input
matrices should be correct.

## Compute Gradients

• Computing the gradients with respect to a variable is so simple.   
• Given a function with input variable x and output variable y   
• To compute dy/dx, simply g = T.grad( y , x )  
• Note: To compute the gradient, y should be a scalar.  
• That’s it!
Example 3
.
If A = [a1 a2; a3 a4]

If B = [b1 b2; b3 b4]

(Note that the dimensions of A and B is not necessary 2 X 2.
Here is just an example.)

C =[a1b1 a2b2; a3b3 a4b4]

D = a1b1 + a2b2 + a3b3 + a4b4

g =[b1 b2; b3 b4]

(line 7)

(line 8)

(line 10)

You cannot compute the gradients of C because it is not a scalar.

Single Neuron
First, let’s implement a neuron
In this stage, let’s assume the model parameters w and b are known

Single Neuron – Shared Variables

• In the last example, a neuron is a function with input x, w and b.
• However, we usually only consider x as input. w and b are model parameters.
• It would be more intuitive if we only have to write “neuron(x)” when using a neuron
• The model parameters w and b still influence neuron(.), but in an implicit way.
• In Theano, the model parameters are usually stored as shared variables.