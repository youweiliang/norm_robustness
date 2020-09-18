import pickle


with open('timing.pkl', 'rb') as f:
    result = pickle.load(f)

for record in result:
    for x in record:
        if x < 1000:
            print(f'{x:.3}\t', end='')
        else:
            print(f'{x:.0f}\t', end='')
    print()