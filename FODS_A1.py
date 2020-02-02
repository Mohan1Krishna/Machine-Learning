import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as pltl
import random

data = pd.read_csv('dataset.csv')
N = len(data)

train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

# Equation: y = w1x1 + w2x2 + w0

X1 = data.iloc[0: train_set_size, 1]
X2 = data.iloc[0: train_set_size, 2]
Y = data.iloc[0: train_set_size, 3]

alpha= 0.00001  # Learning rate


# initialization
w0, w1, w2 = 0, 0, 0
w0_new, w1_new, w2_new = 1, 1, 1

def rms_calc(w1, w2, w0):
    rms_error = 0.0
    for data_index in range(test_set_size):

        index = train_set_size + data_index - 1
        y = data.iloc[index, 3]
        x1 = data.iloc[index, 1]
        x2 = data.iloc[index, 2]

        error = abs(y - ((w1 * x1) + (w2 * x2) + (w0)))
        rms_error += (error * error)

    rms_error /= test_set_size
    rms_error = math.sqrt(rms_error)
    #print("RMS Error:", rms_error)
    return rms_error

def r2_score(w1, w2, w0):
    ss_res = 0.0
    mean=np.mean(data.iloc[:, 3])
    ss_tot=0.0
    for data_index in range(test_set_size):

        index = train_set_size + data_index - 1
        y = data.iloc[index, 3]
        x1 = data.iloc[index, 1]
        x2 = data.iloc[index, 2]

        error = abs(y - ((w1 * x1) + (w2 * x2) + (w0)))
        error1=abs(y-mean)
        ss_res += (error * error)
        ss_tot += (error1 * error1)

    r2=1-(ss_res/ss_tot)
    return r2

def computecost(w1, w2, w0):
    cost = 0.0
    for data_index in range(train_set_size):

        index = data_index 
        y = data.iloc[index, 3]
        x1 = data.iloc[index, 1]
        x2 = data.iloc[index, 2]

        error = abs(y - ((w1 * x1) + (w2 * x2) + (w0)))
        cost += (error * error)

    cost /= test_set_size
    cost = math.sqrt(cost)
    return cost

steps = 0
precision = 0.000001

x_axis = []  # no. of iteration
y_axis = []  # cost asscoiated

while (steps < 100 and (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)) > precision):

    Y_pred = (w1 * X1) + (w2 * X2) + w0

    dr_w0 = (-2 / train_set_size) * sum(Y - Y_pred)
    dr_w1 = (-2 / train_set_size) * sum(X1 * (Y - Y_pred))
    dr_w2 = (-2 / train_set_size) * sum(X2 * (Y - Y_pred))

    w0, w1, w2 = w0_new, w1_new, w2_new

    # new values of parameters
    w0_new = w0 - alpha* dr_w0
    w1_new = w1 - alpha* dr_w1
    w2_new = w2 - alpha* dr_w2

    

    steps += 1


    if steps % 20 == 0:
        x_axis.append(steps)
        y_axis.append(computecost(w1_new, w2_new, w0_new))

print("Final Parameters: ", w0_new, w1_new, w2_new)
print("RMSE: ",rms_calc(w1, w2, w0))
print("R2: ", r2_score( w1_new, w2_new,w0_new))


# graph plotting
pltl.xlabel('Iteration:')
pltl.ylabel('Error:')
pltl.plot(x_axis, y_axis)
pltl.show()


#Normal equation
Y = data.iloc[0: train_set_size, 3]
Y = Y.to_numpy()

X = (data[0: train_set_size].to_numpy())[:, [0, 1, 2]]
X[:, 0] = 1

temp1 = np.dot(X.T, X)
temp2 = np.linalg.inv(temp1)
temp3 = np.dot(X.T, Y)

thetha = np.dot(temp2, temp3)
w0 = thetha[0]
w1 = thetha[1]
w2 = thetha[2]

print(thetha)
#print("Error: ", rms_calc(w1, w2, w0))
print("R2: ", r2_score(w1, w2, w0))

#stochastic gradient descent

precision = 0.000001
epoch = 9

x_axis = []  # iteration
y_axis = []  # error
iter = 0
step_val = 100

data_map = [x for x in range(0, train_set_size)]
for ep in range(epoch):
        steps = 0
        random.shuffle(data_map) 
        while (steps <= step_val):

                index = data_map[steps]
                Y_pred = (w1 * X1[index]) + (w2 * X2[index]) + w0

                dr_w0 = (-2) * (Y[index] - Y_pred)
                dr_w1 = (-2) * (X1[index] * (Y[index] - Y_pred))
                dr_w2 = (-2) * (X2[index] * (Y[index] - Y_pred))

                w0, w1, w2 = w0_new, w1_new, w2_new

                # new values of parameters
                w0_new = w0 - alpha* dr_w0
                w1_new = w1 - alpha* dr_w1
                w2_new = w2 - alpha* dr_w2

                

                steps += 1
                

                if steps % step_val == 0:
                        
                        x_axis.append(iter)
                        iter += step_val
                        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

print("Final Parameters: ", w0_new, w1_new, w2_new)
print("RMSE: ", rms_calc( w1_new, w2_new,w0_new,))
print("R2: ", r2_score(w1_new, w2_new,w0_new))

#Lasso gradient  L1

def lasso_regression():
    steps = 0
    precision = 0.000001
    # initialization
    w0, w1, w2 = 0, 0, 0
    w0_new, w1_new, w2_new = 1, 1, 1

    x_axis = []  # iteration
    y_axis = []  # error
    lambda_vals = [x / 1000 for x in range(1, 10)]

    for lam in lambda_vals:

        # initialization
        w0, w1, w2 = 0, 0, 0
        w0_new, w1_new, w2_new = 1, 1, 1
        steps = 0

        while (steps < steps_count and (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)) > precision):


            Y_pred = (w1 * X1) + (w2 * X2) + w0

            dr_w0 = (-2 / train_set_size) * sum(Y - Y_pred) + (2 * lam * sign(w0))
            dr_w1 = (-2 / train_set_size) * sum(X1 * (Y - Y_pred)) + (2 * lam * sign(w1))
            dr_w2 = (-2 / train_set_size) * sum(X2 * (Y - Y_pred)) + (2 * lam * sign(w2))

            w0, w1, w2 = w0_new, w1_new, w2_new

            # new values of parameters
            w0_new = w0 - alpha* dr_w0
            w1_new = w1 - alpha* dr_w1
            w2_new = w2 - alpha* dr_w2

            

            steps += 1
            

        x_axis.append(lam)
        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

    print("L1 Regularization: Lasso Regression")
    print("Final Parameters: ", w0_new, w1_new, w2_new)
    #print("steps left: ", steps)
    print(rms_calc( w1_new, w2_new,w0_new))
    print(r2_score(w1_new, w2_new,w0_new))

    # graph plotting
    pltl.xlabel('Iteration:')
    pltl.ylabel('Error:')
    pltl.title("L1 Regularization: Lasso Regression")
    pltl.plot(x_axis, y_axis)
    pltl.show()

# Ridge Regression L2

def ridge_regression():
    steps = 0
    precision = 0.000001
    
    x_axis = []  # iteration
    y_axis = []  # error
    lambda_vals = [x / 1000 for x in range(1, 10)]
    for lam in lambda_vals:

        # initialization
        w0, w1, w2 = 0, 0, 0
        w0_new, w1_new, w2_new = 1, 1, 1
        steps = 0

        while (steps < steps_count and (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)) > precision):


            Y_pred = (w1 * X1) + (w2 * X2) + w0

            dr_w0 = (-2 / train_set_size) * sum(Y - Y_pred) + (2 * lam * w0)
            dr_w1 = (-2 / train_set_size) * sum(X1 * (Y - Y_pred)) + (2 * lam * w1)
            dr_w2 = (-2 / train_set_size) * sum(X2 * (Y - Y_pred)) + (2 * lam * w2)

            w0, w1, w2 = w0_new, w1_new, w2_new

            # new values of parameters
            w0_new = w0 - alpha* dr_w0
            w1_new = w1 - alpha* dr_w1
            w2_new = w2 - alpha* dr_w2

            

            steps += 1
            

        x_axis.append(lam)
        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

    print("L2 Regularization: Ridge Regression")
    print("Final Parameters: ", w0_new, w1_new, w2_new)
    #print("steps left: ", steps)
    print(rms_calc(w1_new, w2_new,w0_new))
    print(r2_score( w1_new, w2_new,w0_new))


    # graph plotting
    pltl.xlabel('Iteration:')
    pltl.ylabel('Error:')
    pltl.title("L2 Regularization: Ridge Regression")
    pltl.plot(x_axis, y_axis)
    pltl.show()
ridge_regression()
lasso_regression()
