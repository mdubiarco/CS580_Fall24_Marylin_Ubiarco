import numpy
import matplotlib.pyplot as plt
import pandas 

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

def log_loss(y, y_hat):
    epsilon = 1e-10  
    y_hat = numpy.clip(y_hat, epsilon, 1 - epsilon) 
    return -numpy.mean(y * numpy.log(y_hat) + (1 - y) * numpy.log(1 - y_hat))

# Perceptron Gradient Descent Algorithm
def gradient_descent(X, y, w, b, learning_rate, epochs=100, epsilon=1e-4):
    loss_tracker = []
    boundaries = []  

    for epoch in range(epochs):
        # Calculate predictions
        y_hat = sigmoid(numpy.dot(X, w) + b)

        # Calculate log loss 
        loss = log_loss(y, y_hat)
        loss_tracker.append(loss)

        # Calculate the error
        error = y_hat - y

        # Update b (b + r * error)
        b -= learning_rate * numpy.mean(error)

        # Update w (w_i + r * error * x_i)
        w -= learning_rate * numpy.dot(X.T, error) / len(y)

        if epoch == 0:
            boundaries.append((w, b, 'red', '-'))  
        elif epoch % 10 == 0:
            boundaries.append((w, b, 'green', '--'))

    boundaries.append((w, b, 'black', '-'))

    return w, b, loss_tracker, boundaries

def plot_boundary(X, y, boundaries):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=50)

    # Plot the boundaries
    for w, b, color, linestyle in boundaries:
        Z = sigmoid(numpy.dot(numpy.c_[xx.ravel(), yy.ravel()], w) + b).reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0.5], colors=color, linestyles=linestyle, linewidths=2)

    plt.title("Solution Boundary")
    plt.grid(True)

# Load data from CSV
data = pandas.read_csv('data.csv', sep=',', header=None)
data.columns = ['A', 'B', 'C']


X = data[['A', 'B']].values
y = data['C'].values
w = numpy.random.randn(X.shape[1])
b = numpy.random.randn()  
learning_rate = 0.1
epochs = 100

# Perceptron using Gradient Descent
final_weights, final_bias, loss_tracker, boundaries = gradient_descent(X, y,w, b, learning_rate, epochs)

# Plot the boundary graph and save as gradient_descent.png
plt.figure(figsize=(8, 6))
plot_boundary(X, y, boundaries)
plt.show()
plt.savefig('gradient_descent.png') 
plt.close()

# Plot the log loss (error) graph and save as error_log.png
plt.figure(figsize=(8, 6))
plt.plot(range(0, len(loss_tracker)), loss_tracker, color='b', label='Error Curve')
plt.title("Error Plot")
plt.xlabel("Number of Epochs")
plt.ylabel("Error")
plt.grid(True)
plt.legend()

plt.show()
plt.savefig('error_log.png')
plt.close()

