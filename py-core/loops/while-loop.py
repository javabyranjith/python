magic_number=8
no_of_try = 0
max_try = 3

while no_of_try < max_try:
    input_value = int(input('Guess the number: '))
    no_of_try +=1;
    if input_value == magic_number:
        print('Great! You found...')
        break
else:
    print('Sorry! Wrong guess... :)')

