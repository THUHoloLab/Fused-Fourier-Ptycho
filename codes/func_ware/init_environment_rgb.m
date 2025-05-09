%% parameters
lambda = [0.63123,0.48886,0.4567];
% color_index = 1;
% pix = 2048
lamuda = lambda(color_index);%wavelength um
D_led= 4.0*1000;%Distance between neighboring LED elements
h0 = 65;
h = h0*1000;%Distance between LED and sample
ledMM = 19;
ledNN = 19;%LED array

ledM=ledMM;ledN=ledNN;
k_lamuda=2*pi/lamuda;%wave number

pixel_size = 6.5;%Camera pixel size
mag  = 4;%Magnification
NA = 0.1;%Objective lens numerical aperture
M=pix;N = pix;%Image size captured by CCD
D_pixel=pixel_size/mag;%Image plane pixel size
kmax=NA*k_lamuda;%Maximum wave number corresponding to the numerical aperture of the objective lens

%Magnification of the reconstructed image compared to the original image
MAGimg = 8;%ceil(1+2*D_pixel*3*D_led/sqrt((3*D_led)^2+h^2)/lamuda);%Magnification of the reconstructed image compared to the original image
MM=M*MAGimg;NN=N*MAGimg;%Image size after reconstruction
Niter1 = 50;%Number of iterations
x=-0;
objdx=x*D_pixel;%Location of the small area selected in the sample.As this area becomes larger, the vignetting becomes more pronounced
y=-0;
objdy=y*D_pixel;%
pratio = MAGimg;
%% 频域坐标

[Fx1,Fy1]=meshgrid(-(N/2):(N/2-1),-(M/2):(M/2-1));
Fx1=Fx1./(N*D_pixel).*(2*pi);%Frequency domain coordinates of the original image
Fy1=Fy1./(M*D_pixel).*(2*pi);%Frequency domain coordinates of the original image
Fx2=Fx1.*Fx1;
Fy2=Fy1.*Fy1;
Fxy2=Fx2+Fy2;
Pupil0=zeros(M,N);
Pupil0(Fxy2<=(kmax^2))=1;%Aperture of the objective lens in the frequency domain
[Fxx1,Fyy1]=meshgrid(-(NN/2):(NN/2-1),-(MM/2):(MM/2-1));
Fxx1=Fxx1(1,:)./(N*D_pixel).*(2*pi);%Reconstructing the frequency domain coordinates of an image
Fyy1=Fyy1(:,1)./(M*D_pixel).*(2*pi);%Reconstructing the frequency domain coordinates of an image
%%
dist = 0;
kx = pi/D_pixel*(-1:2/M:1-2/M);
ky = pi/D_pixel*(-1:2/N:1-2/N);
[KX,KY] = meshgrid(kx,ky);

k = 2*pi/lamuda;   % wave number
KX_m = KX;
KY_m = KY;
ind = (KX.^2+KY.^2 >= k^2);
KX_m(ind) = 0;
KY_m(ind) = 0;
 % transfer function
global prop
prop = exp(-1i*dist*sqrt(k^2-KX_m.^2-KY_m.^2));


%% 每个LED在频域对应的像素坐标
% dia_led = 19;% diameter of LEDs used in the experiment
% set up LED coordinates
% h: horizontal, v: vertical
lit_cenv = (ledMM-1)/2;
lit_cenh = (ledMM-1)/2;
vled = (0:ledMM-1) - lit_cenv;
hled = (0:ledMM-1) - lit_cenh;
[hhled,vvled] = meshgrid(hled,vled);
% rrled = sqrt(hhled.^2+vvled.^2);
% LitCoord = rrled<dia_led/2;

k=zeros(ledMM,ledNN);% index of LEDs used in the experiment
for i=1:ledMM
    for j=1:ledNN
        k(i,j)=j+(i-1)*ledNN;
    end
end

% Nled = sum(LitCoord(:));% total number of LEDs used in the experiment

% corresponding angles for each LEDs
v = (-vvled*D_led+objdx)./sqrt((-vvled*D_led+objdx).^2+(-hhled*D_led+objdy).^2+h.^2);%
u = -(-hhled*D_led+objdy)./sqrt((-vvled*D_led+objdx).^2+(-hhled*D_led+objdy).^2+h.^2);%

ttt = u;
u = v;
v = ttt;

% v = (-vvled*D_led+objdx)./sqrt((-vvled*D_led+objdx).^2+(-hhled*D_led+objdy).^2+h.^2);%
% u = -(-hhled*D_led+objdy)./sqrt((-vvled*D_led+objdx).^2+(-hhled*D_led+objdy).^2+h.^2);%
% ttt = u;
% u = v;
% v = ttt;

NAillu=sqrt(u.^2+v.^2);

ledpos_true=zeros(ledMM,ledNN,2);

for i=1:ledMM
    for j=1:ledNN
        Fx1_temp=abs(Fxx1-k_lamuda*u(i,j));
        ledpos_true(i,j,1)=find(Fx1_temp==min(Fx1_temp));
        Fy1_temp=abs(Fyy1-k_lamuda*v(i,j));
        ledpos_true(i,j,2)=find(Fy1_temp==min(Fy1_temp));
    end
end


%% Generate an iterative sequence from the center around the circle outward
ord_ijsum=zeros(ledMM,ledNN);
ord_isum=zeros(1,ledM*ledN);
ord_jsum=zeros(1,ledM*ledN);
ord_ii=(ledMM+1)/2;
ord_jj=(ledNN+1)/2;
ord_isum(1,1)=ord_ii;
ord_jsum(1,1)=ord_jj;
ord_ijsum(ord_ii,ord_jj)=1;
led_num=1;
direction=0;
while (min(min(ord_ijsum))==0)
    led_num=led_num+1;
    direction2=direction+1;
    ord_ii2=round(ord_ii+sin(pi/2*direction2));
    ord_jj2=round(ord_jj+cos(pi/2*direction2));
    if (ord_ijsum(ord_ii2,ord_jj2)==0)
        direction=direction2;
    end
    ord_ii=round(ord_ii+sin(pi/2*direction));
    ord_jj=round(ord_jj+cos(pi/2*direction));
    ord_isum(1,led_num)=ord_ii;
    ord_jsum(1,led_num)=ord_jj;
    ord_ijsum(ord_ii,ord_jj)=1;
end