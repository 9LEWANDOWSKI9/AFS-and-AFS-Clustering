import numpy as np

def get_parameters(data):
    parameters = np.zeros((3, data.shape[1]))

    for j in range(data.shape[1]):
        max_val = float('-inf')
        min_val = float('inf')
        sum_val = 0

        for i in range(data.shape[0]):
            max_val = max(max_val, data[i, j])
            min_val = min(min_val, data[i, j])
            sum_val += data[i, j]
        parameters[0, j] = min_val
        parameters[1, j] = sum_val / data.shape[0]
        parameters[2, j] = max_val

    return parameters

# import numpy as np

# data_matrix = np.array([[1, 2, 3],
#                        [4, 50, 6],
#                        [101, 8, 11]])
#
# result = get_parameters(data_matrix)
#

# print("输入数据矩阵:")
# print(data_matrix)
# print("\n统计参数矩阵:")
# print(result)
#
# # [[  1   2   3]
# #  [  4  50   6]
# #  [101   8  11]]
#
# # [[  1.           2.           3.        ]
# #  [ 35.33333333  20.           6.66666667]
# #  [101.          50.          11.        ]]