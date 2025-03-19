% clc
% clear
pix = 2048;

LED_NUM = [1,8,12,16,24,36,48];

[lambda,...
    CTF_object0,...
    CTF_object,...
    NA,...
    pix_CCD,...
    plane_wave_org,...
    plane_wave_new,...
    df,...
    sample_size,saved_data]=ini_enviroment(LED_NUM,pix);


amplitude = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1];
fqc_shift = [0,4,8,12,16,20];

for shifting_group = 1
    for amplitude_group = 1
        for testing_group = 1

img1 = double(imread('source_image/raw_img.tif'))/255;
% img1 = double(imread('cameraman.tif'))/255;

pratio = 8;
u = img1(:,:,3);
u = imresize(u,[pix*pratio,pix*pratio]);

ground_truth_u = u;
% save('recover/ground_truth.mat', 'ground_truth_u')
% global aberrations
aberrations = zeros(pix);
%{
% rng(2024)
w = rand(1,20);
global idx_pupil pupil_r pupil_theta

for order = 3:18
    aberrations(idx_pupil) = aberrations(idx_pupil) + w(order+1) * ...
    zernfun2(order,pupil_r(idx_pupil),pupil_theta(idx_pupil),'norm');
end

% aberrations = mat2gray(aberrations);
aberrations = mat2gray(aberrations);
aberrations = (aberrations - 0.5) / 0.5;
%}

amplitude_of_aberration = 0;%amplitude(1,amplitude_group) / 2; % 1 for large, 0.01 for small

wavefront2 = (CTF_object0) .*  exp(1i*amplitude_of_aberration*aberrations);
this_datacube = zeros(pix,pix,sum(LED_NUM(:)));
wavefront2_GT = wavefront2;
% imshow(angle(wavefront2_GT),[])
% save pupil_GT wavefront2_GT

shift = fqc_shift(shifting_group);
% figure();imshow(angle(wavefront2),[])
u = fftshift(fft2(u));
for image_count = 1:sum(LED_NUM(:))
    image_count
    len = 1;
    this_datacube(:,:,image_count) = fpm_forward_gen(u, ...
                                                wavefront2, ...
                                                pix * pratio, ...
                                                plane_wave_org(image_count,:), ...
                                                pratio, ...
                                                len,lambda,df,pix_CCD,shift);
    this_datacube(:,:,image_count) = this_datacube(:,:,image_count);
end



this_datacube = this_datacube - min(this_datacube(:));
this_datacube = this_datacube / max(this_datacube(:));

for con = 1:size(this_datacube,3)
    con
    imwrite(uint16((2^16-1) * this_datacube(:,:,con)),"raw_data/b/img" + con + ".tif");
end
% this_datacube = this_datacube + abs(amplitude(1,amplitude_group) * randn(size(this_datacube)));
% saved_data.I_low = this_datacube;
% 
% this_name = ['amp =',num2str(amplitude_group),'_shift=', num2str(shifting_group) ,'_group=',num2str(testing_group)];
% 
% save(['recover/noise_test/',this_name,'.mat'],'saved_data');
        end
    end
end
rmpath(genpath('func_ware'))


function [lambda,...
          CTF_object0,...
          CTF_object,...
          NA,...
          pix_CCD,...
          plane_wave_org,plane_wave_new,df,sample_size,saved_data]=ini_enviroment(LED_NUM,pix)
% clc
% clear
%% Objective properties
lambda = 0.488; % wavelength um
k = 2*pi/lambda;
NA = 0.08; %nurmical aperture



%% CCD properties
Mag = 2;
pix_CCD = pix;
pix_SIZ = 6.5;
sample_size = pix_SIZ/Mag * pix_CCD;

fx_CCD = (-pix_CCD/2:pix_CCD/2-1)/(pix_CCD*pix_SIZ/Mag);
df = fx_CCD(2)-fx_CCD(1);
[fx_CCD,fy_CCD] = meshgrid(fx_CCD);
CTF_CCD = (fx_CCD.^2+fy_CCD.^2)<(NA/lambda).^2;
CTF_object0 = CTF_CCD;

global idx_pupil pupil_r pupil_theta
idx_pupil = ((fx_CCD.^2+fy_CCD.^2)/(NA/lambda).^2) <= 1;
pupil_r = sqrt(fx_CCD.^2 + fy_CCD.^2);
pupil_theta = atan2(fy_CCD,fx_CCD);

ker = fspecial('gaussian',[51,51],3);
noise = mat2gray(imfilter(randn(pix_CCD),ker,'replicate'));

CTF_object = CTF_object0.*exp(1i*0*pi*noise);%;
% imshow((CTF_object),[])

%% LED properties
h_LED = 120; % distance between LED matrix and sample
d_LED =  6; % distance between adjust LED dot


plane_wave_org = zeros(sum(LED_NUM),2); %tilted plane wave
freqXY_calib = zeros(sum(LED_NUM),2);
na_calib = freqXY_calib;
count = 0;
for rings = 1:length(LED_NUM)
    theta = linspace(0,2*pi,LED_NUM(rings) + 1);
    r = d_LED * (rings - 1);
    for con = 1:LED_NUM(rings)
        count = count + 1;

        xy_pos = [r * cos(theta(con)),...
                  r * sin(theta(con)),...
                  0];
        
        v = [0,0,h_LED]-xy_pos;
        v = v/norm(v);

        na_calib(count,2) = -v(1);
        na_calib(count,1) = -v(2);

        plane_wave_org(count,1)= -v(1);
        plane_wave_org(count,2)= -v(2);


        freqXY_calib(count,2) = (plane_wave_org(count,1)/lambda)/df + pix_CCD/2;
        freqXY_calib(count,1) = (plane_wave_org(count,2)/lambda)/df + pix_CCD/2;
    end
end

plane_wave_new = plane_wave_org;


saved_data.na_cal = NA;
saved_data.mag = Mag;
saved_data.dpix_c = pix_SIZ;
saved_data.na_rp_cal = (NA / lambda / df);
saved_data.freqXY_calib = freqXY_calib;
saved_data.na_calib = na_calib;
end

function this_datacube = fpm_forward_gen(wavefront1, ...
                                                wavefront2, ...
                                                pix, ...
                                                plane_wave, ...
                                                pratio, ...
                                                len,lambda,df,pix_CCD,shift)
%% forward inference
ft_wavefront1 = wavefront1;%fftshift(fft2(wavefront1));

for data_con = 1:1
    fxc = round((pix+1)/2+(plane_wave(data_con,1)/lambda)/df + shift*randn(1));
    fyc = round((pix+1)/2+(plane_wave(data_con,2)/lambda)/df + shift*randn(1));
    
    fxl = round(fxc-(pix_CCD-1)/2);fxh=round(fxc+(pix_CCD-1)/2);
    fyl = round(fyc-(pix_CCD-1)/2);fyh=round(fyc+(pix_CCD-1)/2);
    
    F_sub = ft_wavefront1(fyl:fyh,fxl:fxh) .* wavefront2; 
    % sub_wavefront1(:,:,data_con) = F_sub;
end

x = ifft2(ifftshift(F_sub)) / pratio^2;

this_datacube = abs(x);

end

function z = getHOA(num,idx,fr,ft,pix_CCD)


%% aberration 
rng(2020);
co = 2*(rand(1,100)-1);
z = zeros(pix_CCD);
count = 0;
for order = 2:8
    for m = (-order):2:order
        count = count+1;
        z(idx) = z(idx) + co(count)*zernfun(order,m,fr(idx),ft(idx));
        if count == num
            break
        end
    end
    if count == num
        break
    end
end    
end