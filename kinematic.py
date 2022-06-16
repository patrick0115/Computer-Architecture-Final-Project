import math
import numpy as np


# Parameter
a=[0.120 ,0.250, 0.260, 0 ,0 ,0 ]
d=[0 ,0 ,0, 0, 0, 0 ]
alpha=[-90 ,0, 0 ,-90 ,90, 0]
# input
theta=[90, 99 ,-119 ,-10 ,10, 0 ]

# Trigonometric functions
rad = 180 / math.pi
def sin(degree):
	degree = degree / rad
	sin = math.sin(degree)
	return round(sin,15)

def cos(degree):
	degree = degree / rad
	cos = math.cos(degree)
	return round(cos,15)

def tan(degree):
	degree = degree / rad
	tan = math.tan(degree)
	return round(tan,15)

def asin(x):
	asin = math.asin(x) * rad
	return  round(asin,14)

def acos(x):
	acos = math.acos(x) * rad
	return  round(acos,13)

def atan(x):
	atan = math.atan(x) * rad
	return  round(atan,16)
def atan2(x,y):
	atan2 = math.atan2(x,y) * rad
	return  round(atan2,16)


# DH-model
def dh(theta,alpha,d,a):
    A= np.array([[cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta)],
        [sin(theta),cos(theta)*cos(alpha),-cos(theta)*sin(alpha),a*sin(theta)],[0,sin(alpha),cos(alpha),d],[0,0,0,1]])
    # print(T)
    return np.array(A)

# calculate A
A = np.empty((6, 4, 4)) 
T = np.empty((4, 4))
for i, item in enumerate(A):
	A[i] = dh(theta[i], alpha[i], d[i], a[i])
	print('A[',i+1,']:')
	print(A[i])
	
	
# write A in TXT
for i in range(6):
	file_name = 'A'+str(i+1)+'.txt'
	f= open(file_name,'w')
	for j in range(4):
		for k in range(4):
			f.write(str(A[i,j,k]))
			f.write('\n')
f.flush
f.close

# calculate T
T=np.linalg.multi_dot([A[0],A[1],A[2],A[3],A[4],A[5]])

f= open('ANS','w')
for j in range(4):
	for k in range(4):
		f.write(str(T[j,k]))
		f.write('\n')
print('ANS:')
print('T:')
print(T)

phi = atan2(T[1,2],T[0,2])
theta = atan2(cos(phi)*T[ 0, 2] + sin(phi)*T[ 1,2], T[ 2,2])
psi = atan2(-sin(phi)*T[ 0,0]+cos(phi)*T[ 1, 0], -sin(phi)*T[ 0, 1]+cos(phi)*T[ 1, 1]) 

x = T[0,3]
y = T[ 1,3]
z = T[ 2,3] 
print("|         x        |          y        |          z        |        phi        |       theta       |        psi         | ")
print('%19.15f'%x,'%19.15f'%y,'%19.15f'%z,'%19.15f'%phi,'%19.15f'%theta,'%19.15f'%psi,sep="|")







