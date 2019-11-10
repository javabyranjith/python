# convert number to string
firstname = "Ranjith"
age = 34
nameAge = firstname + " " + str(age);
print(nameAge)

print(f"your age: {age}")

animal = "Animal"
animalValue = f"Species is: {animal}"
print(animalValue)

animalValue = "Bird"
print(animalValue)
# to overcome above issue

animal2 = "Elephant"
animalValue2 = "Species is: {}?"  # kind of template
animalValue21 = animalValue2.format(animal2)
print(animal2)
animal2 = "Tiger"
print(animal2)

greetName = "Bal"
greeting = "How are you: {}?"
greetingVal = greeting.format(greetName)
print(greetingVal)