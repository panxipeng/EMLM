#
# import multiprocessing as mp
# import time
# def split_list_n_list(origin_list, n):
#     '''
#     将列表平均分成n段
#     '''
#     if len(origin_list) % n == 0:
#         cnt = len(origin_list) // n
#     else:
#         cnt = len(origin_list) // n + 1
#
#     for i in range(0, n):
#         yield origin_list[i * cnt:(i + 1) * cnt]
#
# def MutiProcessing(kernel_num):
#     def decorator(func):
#         print("print kernel num: ", kernel_num)
#         def infunc(*args, **kwargs):
#             data_list = args[0]
#             task_list = [single_test for single_test in split_list_n_list(data_list, kernel_num)]
#             processes = [mp.Process(target=func, args=args, kwargs=kwargs) for i in range(kernel_num)]
#             # func(*a, *list(kwargs.values()))  #当调用被装饰的函数时没有按照预定顺序以关键词指定参数时会乱序，无法正确匹配
#             for t in processes:
#                 t.start()  # 开始线程或进程，必须调用
#             for t in processes:
#                 t.join()  # 等待直到该线程或进程结束
#         return infunc
#     return decorator
#
# @MutiProcessing(3)
# def fun():
#     for i in range(10):
#         time.sleep(1)
#         print("processing {}, kwargs {}")
#
# task_list = [1, 2, 3]
# fun()


dict1 = {'a': [1, 4], 'b': 2, 'c': 3}
dict2 = {'d': [1, 4], 'e': 2, 'f': 3}
k = 'asd'
v = [1, 2, 3]
print(dict(dict1, **dict(zip(k, v))))
# for item in list(dict.keys()):
#     if isinstance(dict[item], list):
#         print(item)

# la = [1, 2, 3]
# lb = [4, 5, 6]
# lc = [item for item in zip(la, lb)]
# print(lc)

# ll = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# xx = [item for item in zip(*ll)]
# print(xx)