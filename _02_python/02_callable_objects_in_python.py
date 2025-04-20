from dataclasses import dataclass

################################################
# without callable
################################################

# @dataclass
# class Human:
#     name:str
#     age:int

#     def speaks(self):
#         return f"{self.name} is speaking..."
    
#     def eats(self):
#         return f"{self.name} is eating..."
    
# obj1 = Human(name="John", age=25)
# print(obj1.speaks()) # Output: John is speaking...
# print(obj1.eats()) # Output: John is eating...

# obj1() # Output: TypeError: 'Human' object is not callable

################################################
# with callable
################################################

# @dataclass
# class Human:
#     name:str
#     age:int

#     def speaks(self):
#         return f"{self.name} is speaking..."
    
#     def eats(self):
#         return f"{self.name} is eating..."
    
#     def __call__(self):
#         return f"{self.name} is calling..."
    
# obj1 = Human(name="John", age=25)
# print(obj1.speaks()) # Output: John is speaking...
# print(obj1.eats()) # Output: John is eating...

# print(obj1()) # Output: John is calling...

################################################
# Callable Type Hints
################################################

# from typing import Callable

# @dataclass
# class Calculator:
#     operation: Callable[[int, int], str] #takes two integers and returns a string

#     def calculate(self, a: int, b: int) -> str:
#         return self.operation(a, b)

# def add_and_stringify(x: int, y: int) -> str:
#     return str(x + y)

# calc = Calculator(operation=add_and_stringify)
# print(calc.calculate(5, 7))  # Outputs: '12'

################################################
# Refined Example
################################################

from typing import Callable

@dataclass
class Calculator:
    operation: Callable[[int, int], str] #takes two integers and returns a string

    def __call__(self, a: int, b: int) -> str:
        return self.operation(a, b)

def add_and_stringify(x: int, y: int) -> str:
    return str(x + y)

calc = Calculator(operation=add_and_stringify)
print(calc(5, 7))  # Outputs: '12'