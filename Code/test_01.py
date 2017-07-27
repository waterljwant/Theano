import theano
x = theano.tensor.scalar()
y = x**2
f = theano.function([x],y)
print f(-2)
