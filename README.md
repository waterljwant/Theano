# Theano for Deep Learning   
Introduce theano from the beginning, so you can build your own DNN by it.   
Reference: Hung-yi Lee --- [Machine Learning](
http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Theano%20DNN.ecm.mp4/index.html)   

---
## Outline
* Introduction of Theano
* Review Machine Learning
* [01 - Define function](#01---define-function)   
	* [Overview](#overview)   
	* [Step 1. Define Input Variables](#step-1-define-input-variables)
	* [Step 2. Define Output Variables](#step-2-define-output-variables)
	* [Step 3. Declare Function](#step-3-declare-function)
	* [Step 4. Use Function](#step-4-use-function)
* [02 - Compute Gradients](#02---compute-gradients)
* [03 - Single Neuron](#03---single-neuron)
* [04 - Tiny Neural Network](#tiny-neural-network)
---
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
---

## 01 - Define function

### Overview
E.g. Define a function f(x) = x2, then compute f(-2)  
Code: [01_overview.py](https://github.com/waterljwant/Theano/blob/master/Code/01_overview.py#L1-L5)    
```python?linenums
import theano
x = theano.tensor.scalar()
y = x**2
f = theano.function([x],y)
print f(-2)
```
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
So why we define a function by Theano?
**It will be clear when we compute the gradients.**
### Step 1. Define Input Variables
Code: input   
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
Line 7,8,9: let’s print the three variables a, b, c to see what we get
```dos
<TensorType(float32, scalar)>
<TensorType(float32, matrix)>
ming zi
```
a, b, c are symbols without any values

We can give names to variables 
```python?linenums
import theano
x = theano.tensor.scalar('x')
y = theano.tensor.scalar('y')
z = x + y
f = theano.function([x,y],z)
print f(2, 3)
print theano.pp(z)
```
We will get,
```dos
5.0
(x + y)
```

Simplification
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
### Step 2. Define Output Variables
Output variables are defined based on their
relations with the input variables
• Below are some examples
Code: output
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
y1 equals to x1 plus x2.   
y2 equals to x1 times x2.   
y3 is the elementwise multiplication of x3 and x4.   
y4 is the matrix multiplication of x3 and x4.   
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
Code: function   
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

Default value and name for a function
```python?linenums
import theano
import theano.tensor as T
x, y, w = T.scalars('x', 'y', 'w')
z = (x+y)*w
f = theano.function([x,
	theano.In(y, value=1),
	theano.In(w, value=2, name='weights')],
	z)
print f(23, 2, weights=4)
```
### Step 4. Use Function
Line 12: simply use the function f you declared as a normal python function    
Line 13: Print the function output ->

(The theano function output is a numpy.ndarray.)
Examples for Matrix   
Be careful that the dimensions of the input
matrices should be correct.

## 02 - Compute Gradients

• Computing the gradients with respect to a variable is so simple.   
• Given a function with input variable x and output variable y   
• To compute dy/dx, simply g = T.grad( y , x )  
• Note: To compute the gradient, y should be a scalar.  
• That’s it!
Example 1
Code: gradient_example_1
```python?linenums
import theano
import theano.tensor as T
x = T.scalar('x')
y = x ** 2
g = T.grad(y, x)
f = theano.function([x], y)
f_prime = theano.function([x], g)
print f(-2)
print f_prime(-2)
```
Example 2
Code: gradient_example_2
```python?linenums
import theano
import theano.tensor as T
x1 = T.scalar()
x2 = T.scalar()
y = x1 * x2
g = T.grad(y, [x1, x2])
f = theano.function([x1,x2], y)
f_prime = theano.function([x1,x2], g)
print f(2,4)
print f_prime(2,4)
```
Example 3
Code: gradient_example_3
```python?linenums
import theano
import theano.tensor as T
A = T.matrix()
B = T.matrix()
C = A * B
D = T.sum(C)
g = T.grad(D, A)
y_prime = theano.function([A,B], g)
A = [[1,2], [3,4]]
B = [[2,4], [6,8]]
print y_prime(A, B)
```

If A = [a1 a2; a3 a4]

If B = [b1 b2; b3 b4]

(Note that the dimensions of A and B is not necessary 2 X 2.
Here is just an example.)

C = [a1b1 a2b2; a3b3 a4b4]

D = a1b1 + a2b2 + a3b3 + a4b4

g = [b1 b2; b3 b4]

(line 7)

(line 8)

(line 10)

You cannot compute the gradients of C because it is not a scalar.

## 03 - Single Neuron
First, let’s implement a neuron
```python?linenums
import theano
import theano.tensor as T
import random
x = T.vector()
w = T.vector()
b = T.scalar()
z = T.dot(w,x) + b
y = 1/(1+T.exp(-z))
neuron = theano.function(inputs=[x,w,b],outputs=[y], allow_input_downcast=True)
w = [-1,1]
b = 0
for i in range(100):
	print i
	x = [random.random(), random.random()]
	print x
	print neuron(x,w,b)
```	
In this stage, let’s assume the model parameters w and b are known

### Single Neuron – Shared Variables

• In the last example, a neuron is a function with input x, w and b.
• However, we usually only consider x as input. w and b are model parameters.
• It would be more intuitive if we only have to write “neuron(x)” when using a neuron
• The model parameters w and b still influence neuron(.), but in an implicit way.
• In Theano, the model parameters are usually stored as shared variables.
```python?linenums
import theano
import theano.tensor as T
import random
import numpy
x = T.vector()
w = theano.shared(numpy.array([1.,1.]))
b = theano.shared(0.)
z = T.dot(w,x) + b
y = 1/(1+T.exp(-z))
neuron = theano.function(inputs=[x],outputs=[y], allow_input_downcast=True)
print w.get_value()
w.set_value([0.,0.1])
for i in range(100):
	print i
	x = [random.random(), random.random()]
	print x
	print neuron(x)
```
### Single Neuron – Training
Define a cost function C
Then compute ∂C
∂w1
,
∂C
∂w2
, ⋯ ,
∂C
∂wN

and ∂C
```python?linenums
import theano
import theano.tensor as T
import random
import numpy
x = T.vector()
w = theano.shared(numpy.array([-1.,1.]))
b = theano.shared(0.)
z = T.dot(w,x) + b
y = 1/(1+T.exp(-z))
neuron = theano.function(inputs=[x],outputs=[y], allow_input_downcast=True)
y_hat = T.scalar()
cost = T.sum((y-y_hat)**2)
dw,db = T.grad(cost,[w,b])

#--- 01 --- & Gradient Descent --- Tedious
gradient = theano.function(inputs=[x,y_hat],outputs=[dw,db], allow_input_downcast=True)
x = [1, -1]
y_hat = 1
for i in range(100):
	print i
	print neuron(x)
	dw,db = 
	w.set_value(w.get_value() - 0.1*dw)
	b.set_value(b.get_value() - 0.1*db)
	print w.get_value(),b.get_value()
```
Line 31: use the function gradient (defined in the last page) to compute the gradients
Line 32, 33: use the gradients to update the model parameters
```python?linenums
import theano
import theano.tensor as T
import random
import numpy
x = T.vector()
w = theano.shared(numpy.array([-1.,1.]))
b = theano.shared(0.)
z = T.dot(w,x) + b
y = 1/(1+T.exp(-z))
neuron = theano.function(inputs=[x],outputs=[y], allow_input_downcast=True)
y_hat = T.scalar()
cost = T.sum((y-y_hat)**2)
dw,db = T.grad(cost,[w,b])
# Gradient Descent --- Effective
gradient = theano.function(
	inputs=[x,y_hat],
	updates=[(w,w-0.1*dw),(b,b-0.1*db)], 
	allow_input_downcast=True
	)
x = [1, -1]
y_hat = 1
for i in range(100):
	print i
	print neuron(x)
	gradient(x,y_hat)
	print w.get_value(),b.get_value()
```
In deep learning, usually sophisticated update strategy is needed.

What is izip?
https://docs.python.org/2/library/itertools.html#itertools.izip
In this case, you may want to use a function to return
the pair list for parameter update.
Code:
```python?linenums
import theano
import theano.tensor as T
import random
import numpy
from itertools import izip
x = T.vector()
w = theano.shared(numpy.array([-1.,1.]))
b = theano.shared(0.)
z = T.dot(w,x) + b
y = 1/(1+T.exp(-z))
neuron = theano.function(inputs=[x],outputs=[y], allow_input_downcast=True)
y_hat = T.scalar()
cost = T.sum((y-y_hat)**2)
dw,db = T.grad(cost,[w,b])
# Gradient Descent --- Effective
def MyUpdate(paramters,gradients):
	mu = 0.1
	paramters_updates = \
	[(p, p-mu*g) for p,g in izip(paramters,gradients)]
	return paramters_updates

gradient = theano.function(
	inputs=[x,y_hat],
	updates=MyUpdate([w,b],[dw,db])
	)

x = [1, -1]
y_hat = 1
for i in range(100):
	print i
	print neuron(x)
	gradient(x,y_hat)
	print w.get_value(),b.get_value()
```	
## Tiny Neural Network