#This is the learning rate. Higher value may cause divergence
alpha = 0.03

#The data for running gradient descent. The format is data = [[x data], [y data]]
data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.33, 1.41, 1.49, 1.57, 1.65, 1.73, 1.81, 1.89, 1.97, 2.05]]

#Number of both data must be same
if len(data[0]) != len(data[1]):
    raise ValueError

#The number of datasets
m = len(data[0])

#Runs univariate gradientDescent on the given data
def gradientDescent(data):
    #Parameters theta0 and theta1
    theta = [0, 0]
    temp = [10, 10]

    #Set the margin of error as epsilon
    epsilon = 0.00001

    iterations = 0
    converged = False

    #Iterate until convergence
    while not converged:
        iterations += 1
        #calculating the gradients for theta0 and theta1
        grad0 = sum([(theta[0] + theta[1] * data[0][x] - data[1][x])*1 for x in range(m)])
        grad1 = sum([(theta[0] + theta[1] * data[0][x] - data[1][x])*data[0][x] for x in range(m)])

        #First need to set values in temporary variables as we must update theta0 and theta1 simultaneously
        temp[0] = theta[0] - alpha/m * grad0
        temp[1] = theta[1] - alpha/m * grad1

        #check for convergence
        if (abs(temp[0] - theta[0]) < epsilon) and (abs(temp[1] - theta[1]) < epsilon):
            print("iterations = ", iterations)
            converged = True

        #Update theta0 and theta1
        theta[0], theta[1] = temp[0], temp[1]
    return theta

#Driver function to test gradientDescent
def main():
    theta = gradientDescent(data)
    if theta[0] < 0:
        print("%.2fx - %.2f" % (theta[1], -theta[0]))
    else:
        print("%.2fx + %.2f" % (theta[1], theta[0]))

if __name__ == '__main__':
    main()
