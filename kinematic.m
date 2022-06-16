clear all;
close all;

%% Parameter
% DH-model
DtoR = pi / 180; % degree to rad
RtoD = 180 / pi; % rad to degree
a=[0.120 0.250 0.260 0 0 0 ];
d=[0 0 0 0 0 0 0 ];
apha=[-90 0 0 -90 90 0];

%% input
user_input = input('Press "1" for default  \nPress "2" define input by yourself \n');

%% Foreard kinematic


theta=zeros(6);
fprintf('-150<=theta1<=150\n')
fprintf('-30<=theta2<=100\n')
fprintf('-120<=theta3<=0\n')
fprintf('-110<=theta4<=110\n')
fprintf('-180<=theta5<=180\n')
fprintf('-180<=theta6<=180\n')

 n=1;
% let user input the theta\
if user_input == 1
    theta=[90 99 -119 -10 10 0 ];
end
if user_input == 2
    n=1;
    while n<=6
        fprintf('Type theta%d and press enter', n);
        theta(n) = input(':');
        n=n+1;
    end
end
n=1;
while n<=6   
   
    A(:, :, n)=[cosd(theta(n))   -sind(theta(n))*cosd(apha(n))    sind(theta(n))*sind(apha(n))    a(n)*cosd(theta(n));
        sind(theta(n))   cosd(theta(n))*cosd(apha(n))     -cosd(theta(n))*sind(apha(n))   a(n)*sind(theta(n));
        0                sind(apha(n))                    cosd(apha(n))                   d(n) ;
        0                0                                0                               1 ];
    % check the range of theta
    if theta(1) > 150 | theta(1)<-150
        fprintf('theta1 is out of constraint!\n');
        n=n-1;
    end
    if theta(2) > 100 | theta(2)<-30
        fprintf('theta2 is out of constraint!\n');
        n=n-1;
    end
    if theta(3) > 0 | theta(3)<-120
        fprintf('theta3 is out of constraint!\n');
        n=n-1;
    end
    if theta(4) > 110 | theta(4)<-110
        fprintf('theta4 is out of constraint!\n');
        n=n-1;
    end
    if theta(5) > 180 | theta(5)<-180
        fprintf('theta5 is out of constraint!\n');
        n=n-1;
    end
    if theta(6) > 180 | theta(6)<-180
        fprintf('theta6 is out of constraint!\n');
        n=n-1;
    end
    n = n+1;
end


T=A(:, :, 1)*A(:, :, 2)*A(:, :, 3)*A(:, :, 4)*A(:, :, 5)*A(:, :, 6);

% for i =1:6
%     fprintf(' ===============%d================================== \n',i);
%     disp(A(:, :, i));
% end

% phi theta  psi
phi = atan2(T(2, 3), T(1, 3)) * RtoD;
theta = atan2(cos(phi*DtoR)*T(1, 3) + sin(phi*DtoR)*T(2, 3), T(3,3)) * RtoD;
psi = atan2(-sin(phi*DtoR)*T(1, 1)+cos(phi*DtoR)*T(2, 1), -sin(phi*DtoR)*T(1, 2)+cos(phi*DtoR)*T(2, 2)) * RtoD;


% result
disp('   |    n     |     o      |      a      |      p      |');
disp(T);
fprintf('|       x       |       y       |      z         |        phi       |      theta     |        psi        |\n| %f | %f | %f | %f | %f | %f |\n', T(1,4), T(2, 4), T(3, 4), phi, theta, psi);


