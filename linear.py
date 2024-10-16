import pandas
import numpy
import matplotlib.pyplot as plt

##Reads CSV File 
csvFile = pandas.read_csv('linear_regression_data.csv', sep=',', header = None)

##Names the two columns A and B
csvFile.columns = ['A', 'B']

##A and B hold the data for columns A and B
A = csvFile['A']
B = csvFile['B']

##Calculate Means
mean_A = numpy.mean(A)
mean_B = numpy.mean(B)

##Calculate Covariance & Variance
covariance = numpy.mean((A - mean_A) * (B - mean_B))
variance = numpy.mean((A - mean_A) ** 2)

##Calculate Slope, Intercept(b) and Y
m = covariance / variance
b = mean_B - m * mean_A
Y = m * A + b

##Create Graph
plt.scatter(A,B, color= 'black', label= 'Data Points')
plt.plot(A,Y,color= 'blue', label= 'Regression Line')
plt.xlabel('A')
plt.ylabel('B')
plt.title('Linear Regression Data')
plt.legend()
plt.show()

plt.savefig('linear.png')
plt.close()



