command = ""
started = False
while True:
    command = input("Enter the command: ").lower()
    if command == 'start':
        if not started:
            started = True
            print('engine started ----->')
        else:
            print('oops!! engine already started.')
    elif command == 'stop':
        if started:
            started = False
            print('engine stopped <-----')
        else:
            print('oops!! engine already stopped')
    elif command == 'quit':
        break
    else:
        print("Invalid command!!")