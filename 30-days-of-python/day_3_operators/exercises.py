# 30 Days of Python — Day 3: Operators
# Exercises 1-3, 8-20, 23  (the ones that don't ask you questions)
# Run with the play button, or:  python3 exercises.py

import math

print("=" * 50)
print("EXERCISES 1-3: declaring variables")
print("=" * 50)

# 1. Declare your age as an integer variable
age = 26
print("age:", age, type(age))

# 2. Declare your height as a float variable
height = 1.85
print("height:", height, type(height))

# 3. Declare a variable that stores a complex number
complex_number = 3 + 4j
print("complex number:", complex_number, type(complex_number))


print()
print("=" * 50)
print("EXERCISES 8-11: geometry and algebra")
print("=" * 50)

# 8. Slope, x-intercept and y-intercept of y = 2x - 2
# A line written y = mx + b has slope m and y-intercept b, so read them off: m = 2, b = -2
slope_8 = 2
y_intercept = -2                  # the y value when x = 0
x_intercept = 2 / 2               # solve 0 = 2x - 2  ->  x = 1
print("8. slope:", slope_8)
print("8. y-intercept:", y_intercept)
print("8. x-intercept:", x_intercept)

# 9. Slope and Euclidean distance between (2, 2) and (6, 10)
x1, y1 = 2, 2
x2, y2 = 6, 10
slope_9 = (y2 - y1) / (x2 - x1)
distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
print("9. slope:", slope_9)
print("9. euclidean distance:", distance)

# 10. Compare the slopes in tasks 8 and 9
print("10. slopes are equal:", slope_8 == slope_9)

# 11. y = x^2 + 6x + 9, find the x where y = 0
# x^2 + 6x + 9 factors to (x + 3)^2, so y = 0 when x = -3
for x in range(-5, 3):
    y = x ** 2 + 6 * x + 9
    print(f"11. x = {x:>2}  ->  y = {y}")
print("11. y is zero when x = -3")


print()
print("=" * 50)
print("EXERCISES 12-20: comparison and logical operators")
print("=" * 50)

# 12. Length of 'python' and 'dragon', and a falsy comparison
print("12. len('python'):", len("python"))
print("12. len('dragon'):", len("dragon"))
print("12. falsy comparison:", len("python") != len("dragon"))   # False

# 13. Use 'and' to check if 'on' is in both 'python' and 'dragon'
print("13.", "on" in "python" and "on" in "dragon")              # True

# 14. Use 'in' to check if 'jargon' is in the sentence
sentence = "I hope this course is not full of jargon."
print("14.", "jargon" in sentence)                               # True

# 15. There is no 'on' in both dragon and python
print("15.", not ("on" in "dragon" and "on" in "python"))        # False

# 16. Length of 'python', converted to float, then to string
length = len("python")
print("16. int:", length, type(length))
print("16. float:", float(length), type(float(length)))
print("16. string:", str(float(length)), type(str(float(length))))

# 17. Check if a number is even
number = 12
print("17. is", number, "even?", number % 2 == 0)

# 18. Is 7 // 3 equal to int(2.7)?
print("18.", 7 // 3 == int(2.7))                                 # True, both are 2

# 19. Is type('10') equal to type(10)?
print("19.", type("10") == type(10))                             # False: str vs int

# 20. Is int('9.8') equal to 10?
# A trap: int() refuses a string containing a decimal point.
try:
    print("20.", int("9.8") == 10)
except ValueError as error:
    print("20. int('9.8') raises ValueError:", error)
    print("20. workaround, float first:", int(float("9.8")) == 10)   # int(9.8) -> 9, so False


print()
print("=" * 50)
print("EXERCISE 23: the table")
print("=" * 50)

# 23. Display the table: the number, then that number to the powers 0, 1, 2, 3
for i in range(1, 6):
    print(i, i ** 0, i ** 1, i ** 2, i ** 3)
