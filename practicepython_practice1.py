from datetime import date

name = input("What is your name? ")
age = int(input("what is your age? "))

current_date = date.today()
year_to_100 = (current_date.year - age) + 100

print("Hello " + name + ", you will turn 100 years old in ", year_to_100, " years." )

iter_num = int(input("Enter number of iteration: "))

for i in range(iter_num):
    print("Hello " + name + ", \nyou will turn 100 years old in ", year_to_100, " years." )