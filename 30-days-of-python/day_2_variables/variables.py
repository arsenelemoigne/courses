# Day 2: 30 Days of python programming

first_name = "Arsene"
last_name = "le Moigne"
full_name = first_name + last_name

print(full_name)

country = "France"
city = "Paris"
year = 2026
is_married = False
is_true = "happy"
is_light_on = True

a = "1", "2", "3"
print(3*a)

for value in (first_name, full_name, country, city, year, is_married, is_true, a):
    print(value, type(value))
    break


a = len(first_name)
b = len(last_name)

c = b - a 
print(c)

num_one = 5 
num_two = 4

num_sum = num_one + num_two
print(num_sum)

help('keywords')