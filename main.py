import os
import sys

for extension in ['A1', 'A2', 'B1', 'B2']:
    sys.path.append(os.getcwd() + '/' + extension)

import A1_SVM
import A2_SVM
import B1
import B2

# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# Data preprocessing handled in import
# ======================================================================================================================
# Task A1
# model_A1 = A1(args...)                 # Build model object.
# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.
# A1 instantiation run in import
acc_A1_train = str(A1_SVM.train_acc)
acc_A1_test = str(A1_SVM.new_acc)

# ======================================================================================================================
# Task A2
# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
# Clean up memory/GPU etc...
# A2 instantiation run in import
acc_A2_train = str(A2_SVM.train_acc)
acc_A2_test = str(A2_SVM.new_acc)

# ======================================================================================================================
# Task B1
# model_B1 = B1(args...)
# acc_B1_train = model_B1.train(args...)
# acc_B1_test = model_B1.test(args...)
# Clean up memory/GPU etc...
# B1 instantiation run in import
acc_B1_train = str(B1.train_metric[1])
acc_B1_test = str(B1.new_test_metric[1])

# ======================================================================================================================
# Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...
# B2 instantiation run in import
acc_B2_train = str(B2.train_metric[1])
acc_B2_test = str(B2.new_test_metric[1])

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'