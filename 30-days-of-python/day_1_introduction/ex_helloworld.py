#ex 3

my_int = 42
my_float = 37.5
my_string = 'Arthur'
my_bool = True
my_list = ([10, 20, 30])


my_list    = ['a', 'b', 'c']          # ordered, changeable
my_tuple   = ('a', 'b', 'c')          # ordered, frozen
my_set     = {'a', 'b', 'a'}          # unordered, no duplicates -> {'a', 'b'}
my_dict    = {'name': 'Arsène', 'country': 'France'}   # key -> value

for value in [my_int, my_float, my_string, my_bool,
              my_list, my_tuple, my_set, my_dict]:
    print(value, type(value))