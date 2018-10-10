import proj1_helpers
import implementations
import numpy as np

print("Extracting dataset")
y_train, X_train, id1 = proj1_helpers.load_csv_data("train.csv",True)
y_test, X_test, id2 = proj1_helpers.load_csv_data("test.csv",True)
print(X_train.shape)

batch_size = 128

print("Splitting dataset into batch")
X_batch = np.array_split(X_train, int(X_train.shape[0]/batch_size))
y_batch = np.array_split(y_train, int(y_train.shape[0]/batch_size))

print(X_batch[0].shape)

w, _ = implementations.ridge_regression(y_batch[0],X_train[0],1)

print(w)


y_pred = proj1_helpers.predict_labels(w,X_test)

s = 0
tot = 0
for i,y in enumerate(y_pred):
    if y == y_test[i]:
        s += 1
    tot += 1

print(s/tot)
