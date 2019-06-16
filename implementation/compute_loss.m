function [sample_loss] = compute_loss(w_f, samplesf, yf, p)

% Compute the training loss for each sample
nSamples = p.nSamples;
support_sz = numel(w_f{1});

corr_train{1} = bsxfun(@times, permute(w_f{1},[3 1 2]), samplesf{1});
corr_train{2} = bsxfun(@times, permute(w_f{2},[3 1 2]), samplesf{2});

corr_error{1} = bsxfun(@minus, corr_train{1}, permute(yf{1},[3 1 2]));
corr_error{2} = bsxfun(@minus, corr_train{2}, permute(yf{2},[3 1 2]));

error_temp{1} = reshape(corr_error{1},[nSamples, 1, support_sz]);
error_temp{2} = reshape(corr_error{2},[nSamples, 1, support_sz]);
L{1} = 1/support_sz * real(sum(error_temp{1} .* conj(error_temp{1}), 3));
L{2} = 1/support_sz * real(sum(error_temp{2} .* conj(error_temp{2}), 3));

sample_loss = L;