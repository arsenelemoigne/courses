
[]

my_cart = [12, 15, 25, 50]
sum(my_cart)


my_cart.append(100)

print(my_cart)

len(my_cart)

my_cart[2]

my_string = "Hello, world"
my_string[0]

goods = ("one", "burger", "per", "person", "is", "amazing", "sure")
len(goods)
len(my_cart)

goods[1]

products = (my_cart, goods)

my_list = [1, 2, 3, 4, 5]
# "key","value"
my_data = {"name": "Arsene", "last_name": "le Moigne"}


my_data["name"]
my_data.keys()
list(my_data.keys())[0]

my_data["city"] = "Prague"
my_data


my_cart[0]

# for "variable" in "list"
for n in my_cart: 
    #the variable is set here 
    print(n)
    exit

for i in "abc":
    print(i)
    exit

for p in "10": 
    print(p)

for x in range (0,10):
    print(x)

my_data1 = {"name": "Arsene", "last_name": "le Moigne"}
my_data2 = {"name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"}
users_data = [my_data1, my_data2]
print(users_data)

for user in users_data: 
    if "email" in user:
        print(user["email"])


selected_user= {}
my_user_lookup = "Lovelace"

for user in users_data: 
    if "last_name" in user:
        if user ["last_name"] == my_user_lookup:
            selected_user = user

print(selected_user["last_name"])
