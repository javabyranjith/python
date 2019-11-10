# Float
weight = input('Enter your weight (in kg): ')
height = input('Enter your height (in cms): ')
bmi = float(weight) / (int(height) / 100 * int(height) / 100)
print('Your BMI value is: ' + str(bmi))