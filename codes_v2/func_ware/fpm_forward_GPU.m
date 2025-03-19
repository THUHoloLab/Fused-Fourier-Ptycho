%% helper function
function [loss,dldw1,dldw2] = fpm_forward_GPU(wavefront1, ...
                                                wavefront2, ...
                                                kc, ...
                                                b_ledpos, ...
                                                dY_obs, ...
                                                pratio, ...
                                                denoise,type)



loss = 0;
dldw1 = 0*wavefront1;
sub_wavefront1 = dY_obs;

%% denoising constrain
% [s1,c1] = wavedec2(abs(wavefront1),2,'rbio1.1');
% tv_term1 = denoise(1) * (sign(wavefront1).*waverec2(8*exp(-8*abs(s1)) .* sign(s1),c1,'rbio1.1'));
% clear out1;
tv_term1 = 0;

%% forward inference
ft_wavefront1 = fftshift(fft2(wavefront1));
clear wavefront1;
for data_con = 1:size(dY_obs,3)
    kt = kc(1) + b_ledpos(data_con,1);
    kb = kc(2) + b_ledpos(data_con,1);
    kl = kc(3) + b_ledpos(data_con,2);
    kr = kc(4) + b_ledpos(data_con,2);
    sub_wavefront1(:,:,data_con) = ft_wavefront1(kt:kb,kl:kr);
end

x = ifft2_ware(bsxfun(@times,sub_wavefront1,wavefront2),true) / pratio^2;


pp = 1;

[loss,dm] = ret_loss(abs(x).^(pp) - dY_obs.^(pp),'anisotropic',1.02);
% [loss,dm] = ret_frac_loss(abs(x).^(pp),dY_obs.^(pp),'anisotropic');
% x           =   dm .* (x) * pratio^2 .* (abs(x) + 1e-5) .^(pp-2); %
x = bsxfun(@times, dm, sign(x)) * pratio^2;

%% backward propagation

x_record    =   fft2_ware(x,true);
x           =   deconv_pie(x_record,wavefront2,type);

for data_con = 1:size(dY_obs,3)
    kt = kc(1) + b_ledpos(data_con,1);
    kb = kc(2) + b_ledpos(data_con,1);
    kl = kc(3) + b_ledpos(data_con,2);
    kr = kc(4) + b_ledpos(data_con,2);
    dldw1(kt:kb,kl:kr) = dldw1(kt:kb,kl:kr) + x(:,:,data_con);
end
clear x dY_est dY_obs;

dldw1 = ifft2_ware(dldw1,true) + tv_term1;
dldw2 = sum(deconv_pie(x_record,sub_wavefront1,type),3);

end

function out = deconv_pie(in,ker,type)
    switch type
        case 'ePIE'
            out = conj(ker) .* in ./ max(max(abs(ker).^2));
        case 'tPIE'
            fenzi = conj(ker) .* in;
            fenmu = (abs(ker).^2 + 1e-5);
            out = fenzi ./ fenmu;
        case 'none'
            out = bsxfun(@times, conj(ker), in);
        case 'retinex'

            dx = psf2otf([-1,1],[size(in,1),size(in,2)]);
            dy = psf2otf([-1;1],[size(in,1),size(in,2)]);

%             dx = psf2otf([-1,-1,-1;0,0,0;1,1,1],[size(in,1),size(in,2)]);
%             dy = psf2otf([-1,-1,-1;0,0,0;1,1,1]',[size(in,1),size(in,2)]);
            DTD = fftshift(abs(dx).^2 + abs(dy).^2);

            fenzi = conj(ker) .* in .* DTD;
            fenmu = abs(ker).^2 .* DTD + max(max(abs(ker).^2)).*DTD + 1e-5;

            out = fenzi./fenmu;
        otherwise 
            error()
    end

end