import math
import numpy as np

# Parameter
a=[0.120 ,0.250, 0.260, 0 ,0 ,0 ]
d=[0 ,0 ,0, 0, 0, 0, 0 ]
alpha=[-90 ,0, 0 ,-90 ,90, 0]

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

# input
theta=[90, 99 ,-119 ,-10 ,10, 0 ]

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
#	print('A[',i+1,']:')
#	print(A[i])
	
	
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
print(T)
% phi theta  psi
    phi = atan2(T(2, 3), T(1, 3)) * RtoD;
    theta = atan2(cos(phi*DtoR)*T(1, 3) + sin(phi*DtoR)*T(2, 3), T(3,3)) * RtoD;
    psi = atan2(-sin(phi*DtoR)*T(1, 1)+cos(phi*DtoR)*T(2, 1), -sin(phi*DtoR)*T(1, 2)+cos(phi*DtoR)*T(2, 2)) * RtoD;
    
    % result
 %   disp('   |   n   |   o    |    a    |    p    |');
  %  disp(T);
  %  fprintf('|    x     |    y     |    z      |    phi     |   theta   |     psi     |\n| %f | %f | %f | %f | %f | %f |\n', T(1,4), T(2, 4), T(3, 4), phi, theta, psi);








