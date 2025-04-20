################################################
# Without Generics
################################################

# Example without Generics
# def first_element(items):
#     return items[0]

# nums = [1, 2, 3]
# strings = ['a', 'b', 'c']

# num_result = first_element(nums)
# string_result = first_element(strings)

# print(num_result)    # 1
# print(string_result) # 'a'

# Issue: No type checking. We can't restrict or inform about expected data types explicitly.

################################################
# With Generics
# T = TypeVar('T')
################################################

# from typing import TypeVar

# # Type variable for generic typing
# T = TypeVar('T')

# def generic_first_element(items: list[T]) -> T:
#     return items[0]

# nums = [1, 2, 3]
# strings = ['a', 'b', 'c']

# num_result = generic_first_element(nums)        # type inferred as int
# string_result = generic_first_element(strings)  # type inferred as str

# print(num_result)    # 1
# print(string_result) # 'a'

################################################
# Generic Classes
################################################

from typing import Generic, TypeVar, ClassVar
from dataclasses import dataclass, field

# Type variable for generic typing
T = TypeVar('T')

@dataclass
class Stack(Generic[T]): # Generic class
    items: list[T] = field(default_factory=list) # a better way to pass an empty list []
    limit: ClassVar[int] = 30

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self) -> T:
        return self.items.pop()

stack_of_ints = Stack[int]()

print(stack_of_ints) # Stack(items=[])
print(stack_of_ints.limit) # 30
stack_of_ints.push(10)
stack_of_ints.push(20) 
print(stack_of_ints) # Stack(items=[10, 20])
print(stack_of_ints.pop())  # 20

stack_of_strings = Stack[str]()

print(stack_of_strings) # Stack(items=[])
stack_of_strings.push("hello")
stack_of_strings.push("world")
print(stack_of_strings) # Stack(items=['hello', 'world'])

print(stack_of_strings.pop())  # 'world'