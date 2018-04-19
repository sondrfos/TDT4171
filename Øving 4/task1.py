import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(w,x):

  return 1/(1+np.exp(-np.inner(w,x)))

def costFunction(w):

  return (sigmoid(w, [1,0])-1)**2 + (sigmoid(w, [0,1]))**2 + (sigmoid(w, [1,1])-1)**2

def plotting():
  #Make the grid for calculating
  w1 = np.arange(-6,6,0.1)
  w2 = np.arange(-6,6,0.1)
  X, Y = np.meshgrid(w1,w2)

  #Use a double for-loop and the cost-function given in the task to caculate the costs.
  costs = []
  minimum = float("inf")
  for i in range(len(X)):
    costs.append([])
    for j in range(len(Y)):
      costs[i].append(costFunction([X[i,j], Y[i,j]]))

      #find the  minimum by continually comparing to the current minimum
      if costs[i][j] < minimum:
        minimum = costs[i][j]

  #convert to numpy-array for plotting tools to work
  costs = np.array(costs)
  print(minimum)

  #plot the figure
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel("omega_1")
  ax.set_ylabel("omega_2")
  ax.set_zlabel("L_(simple) (omega)")
  ax.plot_wireframe(X, Y, costs)
  plt.show()

  #Can observe minimum at -2.9 and -5.9. It is 0.004976671
 

def costFunctionGradient(w):
    respectToW1 = 2*(sigmoid(w,[1,0]) - 1) * sigmoid(w,[1,0])*(1-sigmoid(w,[1,0])) + 2*(sigmoid(w,[1,1])-1)*sigmoid(w,[1,1])*(1-sigmoid(w,[1,1]))
    respectToW2 = 2*(sigmoid(w,[0,1])) * sigmoid(w,[0,1])*(1-sigmoid(w,[0,1])) + 2*(sigmoid(w,[1,1])-1)*sigmoid(w,[1,1])*(1-sigmoid(w,[1,1]))
    return [respectToW1, respectToW2]

def gradientDescent(w, l, i):
  while(i>0):
    gradient = costFunctionGradient(w)
    w1 = w[0]-l*gradient[0]
    w2 = w[1]-l*gradient[1]
    w = np.append(w1,w2)
    i-=1
  print("New weights:", w)
  print("New cost:", costFunction(w))
  return w, costFunction(w)


def task1():
  l = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
  init_w = [1,1] 
  iterations = 100
  allW = []
  allCosts = []

  for rate in l:
    w,cost = gradientDescent(init_w, rate, iterations)
    allW.append(w)
    allCosts.append(cost)

  allW = np.array(allW)
  allW = allW.transpose()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel("omega_1")
  ax.set_ylabel("omega_2")
  ax.set_zlabel("L_(simple) (omega)")
  ax.plot(allW[0], allW[1], allCosts)
  plt.show()


plotting()