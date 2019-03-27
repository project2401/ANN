import csv
import pickle
import math


def myReader(decoder):
    reader = csv.DictReader(decoder, delimiter=',')
    result = list()
    for row in reader:
        prow = (row['health']+row['weapons']+ row['enemies']+ row['action'])
        result.append(prow)
    return result 

if __name__ == '__main__':
    # with open("task.csv") as myDecoder:
    #     a = myReader(myDecoder)
    # with open('data.pickle', 'wb') as f:
    #     pickle.dump(a, f)
    with open('data.pickle', 'rb') as f:
        data_new = pickle.load(f)
w00 = 0.1        
w01 = 0.8       
w02 = 0.88
w03 = 0.44
w10 = 0.57
w11 = 0.34
w12 = 0.21
w13 = 0.57
w20 = 0.44
w21 = 0.34
w22 = 0.37
w23 = 0.39
ww1 = 0.96
ww2 = 0.15
ww3 = 0.35
ww4 = 0.45

l1 = list(map(int, data_new[0]))
def h1Input(l1, w00, w10, w20):
    h1 = (l1[0]*w00)+(l1[1]*w10)+(l1[2]*w20)
    return 1 / (1 + math.exp(-h1))
print('h1Input = ',h1Input(l1, w00, w10, w20))

def h2Input(l1, w01, w11, w21):
    h2 = (l1[0]*w01)+(l1[1]*w11)+(l1[2]*w21)
    return 1 / (1 + math.exp(-h2))
print('h2Input = ',h2Input(l1, w01, w11, w21))

def h3Input(l1, w02, w12, w22):
    h3 = (l1[0]*w02)+(l1[1]*w12)+(l1[2]*w22)
    return 1 / (1 + math.exp(-h3))
print('h3Input = ',h3Input(l1, w02, w12, w22))

def h4Input(l1, w03, w13, w23):
    h4 = (l1[0]*w03)+(l1[1]*w13)+(l1[2]*w23)
    return 1 / (1 + math.exp(-h4))
print('h4Input = ',h4Input(l1, w03, w13, w23))

h1i = h1Input(l1, w00, w10, w20)
h2i = h2Input(l1, w01, w11, w21)
h3i = h3Input(l1, w02, w12, w22)
h4i = h4Input(l1, w03, w13, w23)

def o1Input ():
    o1 = h1i*ww1+h2i*ww2+h3i*ww3+h4i*ww4
    return o1
print('o1Input = ',o1Input())
o1 = o1Input()


def o1Otput():
    return 1 / (1 + math.exp(-o1))
print('o1Otput = ',o1Otput())

error = o1 - l1[3] 
print("error = ",error)

sigmoidDX = o1*(1 - o1)
print('sigmoidDX = ', sigmoidDX)

weights_delta = error*sigmoidDX
print('weights_delta = ', weights_delta) 

ww1New = ww1 - h1i*weights_delta*0.1
print('ww1New = ',ww1New)

ww2New = ww2 - h2i*weights_delta*0.1
print('ww2New = ',ww2New)

ww3New = ww3 - h3i*weights_delta*0.1
print('ww3New = ',ww3New)

ww4New = ww4 - h4i*weights_delta*0.1
print('ww4New = ',ww4New)

errorForH1 = ww1New*weights_delta
print('errorForH1 = ',errorForH1)
errorForH2 = ww2New*weights_delta
print('errorForH2 = ',errorForH2)
errorForH3 = ww3New*weights_delta
print('errorForH3 = ',errorForH3)
errorForH4 = ww4New*weights_delta
print('errorForH4 = ',errorForH4) 