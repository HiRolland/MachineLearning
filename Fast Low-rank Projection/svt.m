% svt方法求解：
%  Z = argmin||Z||_* + ||Z - W||^2_F
%
%
function [Z, new_sv] = svt(W, n, thr, sv, opt)

    [U, S, V] = lansvd(W, n, n, 1, 'L', opt);
    S = diag(S);
    svp = length(find(S>1/(thr)));
     if svp < sv
        new_sv = min(svp + 1, n);
    else
        new_sv = min(svp + round(0.05*n), n);
    end
    
    if svp>=1
        S = S(1:svp)-1/thr;
    else
        svp = 1;
        S = 0;
    end
    Z = U(:,1:svp)*diag(S)*V(:,1:svp)';
end