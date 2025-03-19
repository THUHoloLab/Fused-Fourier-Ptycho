%% parameters

function [f_pos_set_true,pratio,Pupil0] = init_environment_rgb(color_index, ...
                                                               pix, ...
                                                               led_num, ...
                                                               rot_ang)
global pixel_super

lambda  = [0.660,0.532,0.488]; 
% wavelength
D_led   =  6 * 1000;          % LED distance
H_led   = 120 * 1000;          % LED distance to sample

k_lamuda = 2*pi/lambda(color_index); 

pixel_size  = 6.5 / pixel_super;               % Camera pixel size
mag         = 2 ;               % Magnification
NA          = 0.08 ;             % Objective lens numerical aperture
M = pix * pixel_super;
N = pix * pixel_super;                           % Image size captured by CCD
D_pixel = pixel_size / mag;        % Image plane pixel size
kmax    = NA * k_lamuda;               % Maximum wave number corresponding to the numerical aperture of the objective lens

%Magnification of the reconstructed image compared to the original image
MAGimg = 8;              % ceil(1+2*D_pixel*3*D_led/sqrt((3*D_led)^2+h^2)/lamuda);%Magnification of the reconstructed image compared to the original image
MM  =   M*MAGimg;
NN  =   N*MAGimg;        % Image size after reconstruction
pratio = MAGimg;
led_total = sum(led_num(:));

pix_large = pix * pixel_super;
%% spatial frequency
fx_CCD = (-pix_large/2:pix_large/2-1)/(pix_large * D_pixel);
df = fx_CCD(2)-fx_CCD(1);
[fx_CCD,fy_CCD] = meshgrid(fx_CCD);
CTF_CCD = (fx_CCD.^2+fy_CCD.^2)<(NA/lambda(color_index)).^2;
Pupil0 = CTF_CCD;
% figure();
% imshow(CTF_CCD,[]);
% title("NA = " + NA + "     X" + mag)
Rcam = lambda(color_index) / NA*mag / 2 /pixel_size;
RLED = NA*sqrt(D_led^2+H_led^2)/D_led;
Roverlap = 1/pi*(2*acos(1/2/RLED)-1/RLED*sqrt(1-(1/2/RLED)^2));

disp(['the overlapping rate is ',num2str(Rcam)]);
disp(['the overlapping rate is ',num2str(Roverlap)]);
plane_wave_org = zeros(led_total,2); %initial non-shifted plane wave

%% plane wave direction
count = 0;
for ring = 1:length(led_num)
    phi = linspace(0,2*pi,led_num(ring)+1) + rot_ang;
    for con = 1:led_num(ring)
        count = count + 1;
        r = D_led * (ring - 1);
        v = [0,0,H_led]-[r .* cos(phi(con)),r .* sin(phi(con)),0];
        v = v/norm(v);
        plane_wave_org(count,1) = -v(1);
        plane_wave_org(count,2) = -v(2);  
    end
end

f_pos_set_true = zeros(led_total,4);
for con = 1:led_total
    fxc = round((MM+1)/2 + (plane_wave_org(con,1)/lambda(color_index))/df);
    fyc = round((MM+1)/2 + (plane_wave_org(con,2)/lambda(color_index))/df);
    
    fxl = round(fxc-(pix_large-1)/2);fxh=round(fxc+(pix_large-1)/2);
    fyl = round(fyc-(pix_large-1)/2);fyh=round(fyc+(pix_large-1)/2);
    f_pos_set_true(con,:) = [fxl,fxh,fyl,fyh];
end



