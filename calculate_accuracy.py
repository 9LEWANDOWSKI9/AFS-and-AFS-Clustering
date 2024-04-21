def calculate_accuracy(result):
    print("==========输出聚类结果==========")
    accuracy = 0
    wrong = 0
    right = 0

    for i, r in enumerate(result):
        rt = r["right"]
        rw = r["wrong"]
        right += r["right"]
        wrong += r["wrong"]
        print(f"==========第 {i} 类共有 {rw + rt} 个==========")
        print(f"其中正确 {rt} 个, 错误 {rw} 个")

    accuracy = right / (right + wrong)
    print(f"准确率为 {accuracy}")


# result_data = [{"right": 20, "wrong": 5}, {"right": 15, "wrong": 3}]
# calculate_accuracy(result_data)
