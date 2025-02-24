function [loss,dldw1,dldw2] = fpm_forward_fused(wavefront1, ...
                                                wavefront2, ...
                                                kc, ...
                                                b_ledpos, ...
                                                dY_obs, ...
                                                pratio, ...
                                                len, ...
                                                denoise,type)



loss = 0;
% dldw1 = 0*wavefront1;

%% forward inference
kt = kc(1) + b_ledpos(:,1);
kl = kc(3) + b_ledpos(:,2);

ledIdx = int32(gpuArray([kl';kt']));

[dldw1,dldw2] = fullyfusedFPM(wavefront1,wavefront2,dY_obs,ledIdx,pratio);

% for data_con = 1:len
%     kt = kc(1) + b_ledpos(data_con,1);
%     kb = kc(2) + b_ledpos(data_con,1);
%     kl = kc(3) + b_ledpos(data_con,2);
%     kr = kc(4) + b_ledpos(data_con,2);
%     dldw1(kt:kb,kl:kr) = dldw1(kt:kb,kl:kr) + x(:,:,data_con);
% end

% dldw1 = ifft2(ifftshift(ifftshift(dldw1,1),2));
% dldw2 = sum(deconv_pie(x_record,sub_wavefront,type),3);

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
            out = bsxfun(@times,conj(ker),in);
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

function out = get_tv(temp_o)

    mask = imfilter(temp_o,ones(3)/9,'replicate');
    mask(mask>1) = 1;

    mask = mask.^3;
%     mask = (mask>0.5);
    x = imfilter(temp_o,[-1,1],'replicate');
    y = imfilter(temp_o,[-1;1],'replicate');
    x = 8*exp(-8*abs(x)).*sign(x);
    y = 8*exp(-8*abs(y)).*sign(y);

    out = imfilter(x,[1,-1,0],'replicate') +...
          imfilter(y,[1;-1;0],'replicate');

    out = out .* mask;
end

function out = cos_loss(x,y)
fenzi = sum(sum(x.*y));
fenmu = sum(sum(abs(x).*abs(y)));
out = fenzi ./ (fenmu + 1e-5);

end