
import random

# 從 1 到 49 中隨機抽出 7 個不重複的數字
numbers = random.sample(range(1, 50), 7)

# 依數字大小排序（方便查看）
numbers.sort()

print("抽出的 7 個數字為：", numbers)
