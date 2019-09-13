from sklearn.preprocessing import MinMaxScaler, StandardScaler

x = [[1, 2, 3],
     [3, 2, 1],
     [5, 7, 3],
     [4, 9, 8],
     [8, 5, 2],
     [3, 4, 8],
     [1, 7, 9],
     [7, 7, 4],
     [2, 6, 3],
     [4, 4, 5]]


print(x)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

print("normalized matrix: ")
print(x)