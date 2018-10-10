import proj1_helpers
import implementations

print("coucou")

y_train, X_train, id1 = proj1_helpers.load_csv_data("train.csv")
y_test, X_test, id2 = proj1_helpers.load_csv_data("test.csv")

w, _ = implementations.least_squares(y_train,X_train)

print(w)


y_pred = proj1_helpers.predict_labels(w,X_test)

s = 0
tot = 0
for i,y in enumerate(y_pred):
    if y == y_test[i]:
        s += 1
    tot += 1

print(s/tot)
