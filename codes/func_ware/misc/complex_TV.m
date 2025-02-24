function o = complex_TV(S, lambda,type)

betamax = 1e4;

fx = [1, -1];
fy = [1; -1];
[N,M,D] = size(S);

otfFx = psf2otf(fx,[N,M]);
otfFy = psf2otf(fy,[N,M]);

Normin1 = fft2(S);

Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
if D>1
    Denormin2 = repmat(Denormin2,[1,1,D]);
end
beta = lambda;

o = S;

foo = @(x) 1 * exp(-1*abs(x));

while beta < betamax
    lambeta = lambda/beta;
    Denormin   = 1 + beta*Denormin2;
    % h-v subproblem
    u = [diff(o,1,2), o(:,1,:) - o(:,end,:)];
    v = [diff(o,1,1); o(1,:,:) - o(end,:,:)];

    switch type
        case 'isotropic'
            den = sqrt(u.^2+v.^2) + 1e-5;
            u = u ./abs(den) .* max(abs(den) - lambeta, 0);
            v = v ./abs(den) .* max(abs(den) - lambeta, 0);
        case 'anisotropic'
            u = sign(u) .* max(abs(u) - lambeta ,0);
            v = sign(v) .* max(abs(v) - lambeta ,0); %.* foo(abs(v)) / foo(lambeta)
        case 'hard'
            den = u.^2 + v.^2;
            u = (abs(den) > lambeta) .* u;
            v = (abs(den) > lambeta) .* v;
        otherwise
            error('the type should be isotropic, or anisotrapic, or hard');
    end
    

    % o subproblem
    Normin2 = [u(:,end,:) - u(:, 1,:), -diff(u,1,2)];
    Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
    Fo = (Normin1 + beta*fft2(Normin2))./Denormin;
    o = (ifft2(Fo));

    beta = beta*2;

end

end

