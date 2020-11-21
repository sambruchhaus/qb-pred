# qb_pred_knn.py
# Uses k-nearest neighbor to predict the DVOA of a QB using his college Completion% and Games Played
# Author: Sam Bruchhaus

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Read text files, using the numpy.loadtxt() function.
def load_file(file_name):
    return np.loadtxt(file_name)


# Multiclass k-nearest neighbor classifier.
def knn(test_points, train_points, k):
    yis = np.empty([0])
    # For each test point xi
    for xi in test_points:
        dijs = np.empty([0])
        # For each training point xj
        for xj in train_points:
            # Compute dij = |xi-xj|^2 where | | is the Euclidean distance
            dij = np.linalg.norm(xi[1:3] - xj[1:3]) * 2
            dijs = np.append(dijs, dij)
        # Sort the training point according to dij's
        to_sort = np.c_[train_points, dijs]
        sorted = to_sort[np.argsort(to_sort[:, 3])]
        # Predict yi = most frequent labels of k nearest neighbors
        nearest = sorted[0:k]
        yi = np.average(nearest[:, 0])
        yis = np.append(yis, yi)
    return yis


# Choose the best k value from k=1,11,21,31  by 10-fold cross validation.
def crossval_knn(train_points):
    # For k in [2-7]
    ks = np.arange(1, 8)
    k_errors = np.empty([0])
    for k in ks:
        split_errors = np.empty([0])
        # Repeat over 10 splits
        for i in range(10):
            # use 90% as training
            sample_train = np.empty([0, train_points.shape[1]])
            indexes_train = np.random.choice(train_points.shape[0], (int(train_points.shape[0] * 0.9)), replace=False)
            for index_train in indexes_train:
                sample_train = np.append(sample_train, np.array([train_points[index_train]]), axis=0)
            # 10% as testing
            sample_test = np.empty([0, train_points.shape[1]])
            indexes_test = (np.setdiff1d(np.arange(len(train_points)), indexes_train))
            for index_test in indexes_test:
                sample_test = np.append(sample_test, (np.array([train_points[index_test]])), axis=0)
            # For each split, apply k-nn
            sample_ys = knn(sample_test, sample_train, k)
            # measure error
            for x in range(len(sample_ys)):
                error = (sample_ys[x] - sample_test[x, 0]) ** 2
                split_errors = np.append(split_errors, error)
        avg_split_error = np.average(split_errors)
        k_errors = np.append(k_errors, avg_split_error)
    print(k_errors)
    return ks[np.where(k_errors == k_errors.min())[0][0]]


# load files
train = load_file("qb_pred_data.train")
test = load_file("qb_pred_data.test")
# Report those cross validation error for each k.
best_k = crossval_knn(train)
print(best_k)
# For the best value of k,  run k-nn again using all training data.
best_k_results = knn(test, train, best_k)
print(best_k_results)

X = np.vstack((np.ones(len(train)), train[:,[1,2]].T)).T
reg_Y_input = np.sign(train[:,0] + 10).transpose()
Xdag = np.matmul( np.linalg.pinv(np.matmul(X.transpose(), X)), X.transpose() )
w_reg = np.matmul(Xdag, reg_Y_input)
if (w_reg[2] != 0):
    m_reg = -(w_reg[0] / w_reg[2]) / (w_reg[0] / w_reg[1])
else:
    m = 0
if w_reg[2] != 0:
    b_reg = -w_reg[0]/w_reg[2]
else:
    b_reg = 0
yreg_coordinates = m_reg* np.arange(23, 52) + b_reg
plt.plot(np.arange(23,52), yreg_coordinates, 'm')

# Plot True Results
colors_true = []
for y in train[:, 0]:
    if y < -10:
        colors_true.append("#FF3333")
    elif -10 <= y < 0:
        colors_true.append("#FFC133")
    elif 0 <= y < 10:
        colors_true.append("#86FF33")
    elif 10 <= y < 30:
        colors_true.append("#33FFEC")
    else:
        colors_true.append("#090C05")
plt.figure(1)
plt.title("Pro Only Results")
plt.xlabel('Games Played')
plt.ylabel('Completion Percentage')
f = open("qb_pred_codes_college.train", 'r')
player_codes_pro = f.readlines()
f.close()
for i in range(len(player_codes_pro)):
    plt.annotate(player_codes_pro[i], xy=(train[i, 1], train[i, 2]), size='4.5')

plt.scatter(train[:, 1], train[:, 2], c=colors_true, s=13)

zero = mpatches.Patch(color='#FF3333', label='Benched')
one = mpatches.Patch(color='#FFC133', label='Below Average Starter')
two = mpatches.Patch(color='#86FF33', label='Above Average Starter')
three = mpatches.Patch(color='#33FFEC', label='Quality Starter')
nine = mpatches.Patch(color='#090C05', label='MVP')
plt.legend(handles=[zero, one, two, three, nine], loc='best', prop={'size': 10})

# Plot Pred Results
colors_pred = []
for y in best_k_results:
    if y < -10:
        colors_pred.append("#FF3333")
    elif -10 <= y < 0:
        colors_pred.append("#FFC133")
    elif 0 <= y < 10:
        colors_pred.append("#86FF33")
    elif 10 <= y < 30:
        colors_pred.append("#33FFEC")
    else:
        colors_pred.append("#090C05")
plt.figure(2)
plt.title("Pro & College Results")
plt.xlabel('Games Played')
plt.ylabel('Completion Percentage')
f = open("qb_pred_codes_college.test", 'r')
player_codes_pred = f.readlines()
f.close()
for i in range(len(player_codes_pro)):
    plt.annotate(player_codes_pro[i], xy=(train[i, 1], train[i, 2]), size='4.5')

for i in range(len(player_codes_pred)):
    plt.annotate(player_codes_pred[i], xy=(test[i, 1], test[i, 2]), size='4.5')

plt.scatter(train[:, 1], train[:, 2], c=colors_true, s=13)
plt.scatter(test[:, 1], test[:, 2], c=colors_pred, s=13)
plt.plot(np.arange(23,52), yreg_coordinates, 'm')

zero = mpatches.Patch(color='#FF3333', label='Benched')
one = mpatches.Patch(color='#FFC133', label='Below Average Starter')
two = mpatches.Patch(color='#86FF33', label='Above Average Starter')
three = mpatches.Patch(color='#33FFEC', label='Quality Starter')
nine = mpatches.Patch(color='#090C05', label='MVP')
plt.legend(handles=[zero, one, two, three, nine], loc='best', prop={'size': 10})

plt.show()
