import numpy 
import matplotlib.pyplot as plt
import pandas 

# Load data
data = pandas.read_csv('data.csv', sep=',', header=None)
data.columns = ['A','B','C']

X = data[['A', 'B']].values
y = data['C'].values

# Initialize weights and bias randomly
numpy.random.seed(42)
w = numpy.random.randn(X.shape[1])  
b = numpy.random.randn(1)

# Learning rate
r = 0.1

# Classification function 
def classify(X, w, b):
    return numpy.sign(numpy.dot(X, w) + b)

# Rule for misclassified points
def update_weights(X, y, w, b, r):
    for i in range(len(X)):
        prediction = classify(X[i], w, b)
        if prediction != y[i]:  
            if y[i] == 1:  
                w += r * X[i]
                b += r
            else:  
                w -= r * X[i]
                b -= r
    return w, b


def perceptron(X, y, w, b, r, epochs=10):
    for epoch in range(epochs):
        w, b = update_weights(X, y, w, b, r)
        plot_boundary(X, y, w, b, epoch, epochs)  
    return w, b

# Plotting the boundary
def plot_boundary(X, y, w, b, epoch, epochs):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))
    zz = -(w[0] * xx + w[1] * yy + b) / w[1]  

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o', edgecolor='k')

    # Plot the boundary
    if epoch == 0:
        plt.contour(xx, yy, zz, levels=[0], colors='red', linewidths=2, linestyles='--')  # Initial line in red
    elif epoch == epochs - 1:
        plt.contour(xx, yy, zz, levels=[0], colors='black', linewidths=2)  # Last line in black
    else:
        plt.contour(xx, yy, zz, levels=[0], colors='green', linewidths=1, linestyles='--')  # Subsequent lines in green

    
    plt.title('Solution Boundary')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

# Run the perceptron training
final_weights, final_bias = perceptron(X, y, w, b, r, epochs=10)

# Show the plot and save it 
plt.show()
plt.savefig('perceptron.png') 
plt.close()


