class ClassA:
    def __init__(self): self.__var = 10

    def method1(self): return self.get_var() + 1

    def get_var(self): return self.__var
    def set_var(self, val): self.__var = val


class ClassB(ClassA):
    def __init__(self): self.__var = 20

    def get_var(self): return self.__var
    def set_var(self, val): self.__var = val


ObjectA = ClassA()
ObjectB = ClassB()

print(ObjectA.method1())
print(ObjectB.method1())