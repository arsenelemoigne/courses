# 30 Days of Python — Day 3: Operators
# Exercises 4, 5, 6, 7, 21, 22 — the ones that ask you questions.
# This file PAUSES: type a number, press Enter, repeat.

print("--- 4. Area of a triangle ---")
base = float(input("Enter base: "))
height = float(input("Enter height: "))
print("The area of the triangle is", 0.5 * base * height)

print()
print("--- 5. Perimeter of a triangle ---")
side_a = float(input("Enter side a: "))
side_b = float(input("Enter side b: "))
side_c = float(input("Enter side c: "))
print("The perimeter of the triangle is", side_a + side_b + side_c)

print()
print("--- 6. Area and perimeter of a rectangle ---")
length = float(input("Enter length: "))
width = float(input("Enter width: "))
print("Area of the rectangle is", length * width)
print("Perimeter of the rectangle is", 2 * (length + width))

print()
print("--- 7. Area and circumference of a circle ---")
pi = 3.14
radius = float(input("Enter radius: "))
print("Area of the circle is", pi * radius * radius)
print("Circumference of the circle is", 2 * pi * radius)

print()
print("--- 21. Weekly pay ---")
hours = float(input("Enter hours: "))
rate = float(input("Enter rate per hour: "))
print("Your weekly earning is", hours * rate)

print()
print("--- 22. Seconds lived ---")
years = int(input("Enter number of years you have lived: "))
print("You have lived for", years * 365 * 24 * 60 * 60, "seconds.")
