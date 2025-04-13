################################################
# Traditional Classes
################################################

# class Dog:
#     # (constructor)
#     def __init__(self, name):
#         self.name = name

#     # method
#     def bark(self):
#         print(f"{self.name} says Woof!")

#     # string representation of the object
#     def __repr__(self):
#         return f"Dog(name='{self.name}')"

# # Creating an object of the class
# my_dog = Dog("Buddy")

# # Accessing method
# my_dog.bark()  # Output: Buddy says Woof!

# # Accessing property (__repr__)
# print(my_dog) # Output: Dog(name='Buddy')

# Dog is a class.
# my_dog is an object (or instance) of that class.
# name is a property/attribute.
# bark() is a method.

################################################
# @staticmethod in Python
################################################

# A static method doesn’t depend on class instance (self) or class (cls). 
# It’s just a regular function inside a class — grouped logically but doesn't operate on the class/object directly.
# class Math:
#     @staticmethod
#     def add(a, b):
#         return a + b

# # Can call without creating object
# print(Math.add(3, 4))  # Output: 7

################################################
# @dataclasses
################################################

from dataclasses import dataclass
from typing import ClassVar

@dataclass
class American:
# class variables
  national_language: ClassVar[str] = "English"
  national_food: ClassVar[str] = "Hamburger"
  normal_body_temperature: ClassVar[float] = 98.6
# instance variables
  name: str
  age: int
  weight: float
  liked_food: str

# instance methods
  def speaks(self):
    return f"{self.name} is speaking... {American.national_language}"

  def eats(self):
    return f"{self.name} is eating..."

# static method
  @staticmethod
  def country_language():
    return American.national_language
  
# class method
  @classmethod
  def cultural_info(cls):
      return f"Language: {cls.national_language}, Food: {cls.national_food}, Body Temp: {cls.normal_body_temperature}"
  
# invoking the static method
print(American.country_language())
# output= English

# invoking the class method
print(American.cultural_info())
# output= Language: English, Food: Hamburger, Body Temp: 98.6

# Creating an instace of the class
john = American(name="John", age=25, weight=65, liked_food="Pizza")

# invoking the instace methods
print(john.speaks()) # Output: John is speaking... English
print(john.eats()) # Output: John is eating...

print(john) # Output: American(name='John', age=25, weight=65, liked_food='Pizza')
print(john.name) # Output: John
print(john.age) # Output: 25
print(john.weight) # Output: 65
print(American.national_language) # Output: English

# self referes to the instance of the class
# cls refers to the class itself