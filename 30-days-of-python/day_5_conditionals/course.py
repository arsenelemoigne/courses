# Day 5: Conditionals
# Write code here. Save with Cmd+S. Run with the ▷ button, top right.

print("--- day 5 ---")

if True:
    print("hi")

if False:
    print("hi")

if not False:
    print("hi")

True == False
10 > 5 

str(True).lower() == "true" 
str("true").title() == "True"

abc = [1,2,3,4,5]
abc_sq = []

for nb in abc: 
    sq_nb = nb ** 2
    abc_sq.append(sq_nb)

print(abc_sq)

is_even = []
is_odd = []

for nb in abc_sq:
    if nb %2 == 0:
        print("nb is even")
        is_even.append(nb)

    else:
        is_odd.append(nb)

print("even:", is_even)
print("odd:", is_odd)


x = 10 
i = 0 

while i < x:
    print(i)
    i = i + 1


x = 10 
i = 0 
z = 12

while i < x:
    z = z * 2 
    print(z)
    if z > 150:
        break

print(z)