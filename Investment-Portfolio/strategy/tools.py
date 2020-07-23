import numpy as np
import math
import random


def getRandomN(n, N):
    """

    :param n: 随机数个数
    :param N: 范围从0到N
    :return:
    """
    choises = set()
    while len(choises) < n:
        choises.add(random.randint(0, N - 1))
    return np.array(list(choises))

