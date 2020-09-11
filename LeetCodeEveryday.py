# 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
# candidates 中的数字可以无限制重复被选取。
# 说明：
# 所有数字（包括 target）都是正整数。
# 解集不能包含重复的组合。 
# 示例 1：
# 输入：candidates = [2,3,6,7], target = 7,
# 所求解集为：
# [
#   [7],
#   [2,2,3]
# ]
# 示例 2：
# 输入：candidates = [2,3,5], target = 8,
# 所求解集为：
# [
#   [2,2,2,2],
#   [2,3,3],
#   [3,5]
# ]
#
# 提示：
# 1 <= candidates.length <= 30
# 1 <= candidates[i] <= 200
# candidate 中的每个元素都是独一无二的。
# 1 <= target <= 500
import math
def combinationSum(candidates, target):
    res = []

    def walk(start, subseq):
        for i in range(start, len(candidates)):
            tempsub = subseq.copy()
            tempsub.append(candidates[i])
            if math.fsum(tempsub) < target:
                walk(i, tempsub)
            elif math.fsum(tempsub) == target:
                res.append(tempsub)

    walk(0, [])
    return res

# candidates 中的每个数字在每个组合中只能使用一次。
# 示例 1:
# 输入: candidates = [10,1,2,7,6,1,5], target = 8,
# 所求解集为:
# [
#   [1, 7],
#   [1, 2, 5],
#   [2, 6],
#   [1, 1, 6]
# ]
# 示例 2:
# 输入: candidates = [2,5,2,1,2], target = 5,
# 所求解集为:
# [
#   [1,2,2],
#   [5]
# ]
def combinationSum2(candidates, target):
    res_temp = set()

    def walk(start, remain, subseq):
        for i in range(start, len(remain)):
            remain_copy = remain.copy()
            tempsub = subseq.copy()
            tempsub.append(remain_copy.pop(i))
            if math.fsum(tempsub) < target:
                walk(0, remain_copy, tempsub)
            elif math.fsum(tempsub) == target:
                tempsub.sort()
                res_temp.add(tuple(tempsub))

    walk(0, candidates, [])
    return res_temp


# 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
# 说明：
# 所有数字都是正整数。
# 解集不能包含重复的组合。 
# 示例 1:
# 输入: k = 3, n = 7
# 输出: [[1,2,4]]
# 示例 2:
# 输入: k = 3, n = 9
# 输出: [[1,2,6], [1,3,5], [2,3,4]]
def combinationSum3(k, n):
    orig = list(range(1, 10))
    res = []

    def walk(remain, subres):
        if len(remain) + len(subres) < k:
            return
        for i in range(len(remain) - k + len(subres) + 1):
            tempre = remain[i:].copy()
            tempsub = subres.copy()
            tempsub.append(tempre.pop(0))
            if len(tempsub) == k:
                if math.fsum(tempsub) == n:
                    res.append(tempsub)
            else:
                walk(tempre, tempsub)

    walk(orig, [])
    return res

combinationSum3(3, 7)


def combine(n, k):
    orig = list(range(1, n+1))
    res = []

    def walk(remain, subres):
        if len(remain) + len(subres) < k:
            return
        for i in range(len(remain) - k + len(subres) + 1):
            tempre = remain[i:].copy()
            tempsub = subres.copy()
            tempsub.append(tempre.pop(0))
            if len(tempsub) == k:
                res.append(tempsub)
            else:
                walk(tempre, tempsub)

    walk(orig, [])
    return res


