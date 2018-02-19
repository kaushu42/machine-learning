#This is the training set that will be used for regression
#Based on this data, the program will fit the best straight line for the data
training_set = [[1, 15], [2, 46], [3, 12], [4, 48], [5, 78]]
m = len(training_set)

#We have to find a and b using gradient descent
#Define the hypothesis function.
def h(x, a, b):
	return a + b*x 

#This is the cost function. Our main objective is to reduce this cost
def J(a, b):
	ss = 0 #Sum of square of errors
	for data in training_set:
		temp = (h(data[0], a, b) - data[1])
		ss += temp*temp 
	return 1/(2*m)*ss

#This is partial derivative of cost function with respect to a
def daJ(a, b):
	s = 0
	for data in training_set:
		s += h(data[0], a, b) - data[1]
	return 1/m*s

#This is partial derivative of cost function with respect to b
def dbJ(a, b):
	s = 0
	for data in training_set:
		s += (h(data[0], a, b) - data[1])*data[0]
	return 1/m*s

#The gradient descent algorithm
def gradientDescent():
	learning_rate = 0.1
	a = 0
	b = 0
	for i in range(1000):
		aTemp = a - learning_rate * daJ(a, b) #Temporary variables required as both 'a' and 'b' must be
		bTemp = b - learning_rate * dbJ(a, b) #updated simultaneously.
		a = aTemp
		b = bTemp
	return a, b

#Test function for the algorithm
def main():
	a, b = gradientDescent()
	print("{:.4f}x + {:.4f}".format(b, a))

if __name__ == '__main__':
	main()



