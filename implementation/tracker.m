function results = tracker(p, im, bg_area, fg_area, area_resize_factor)

nDim = p.nDim;
feat_type = p.feat_type;
layerInd = p.layerInd;

lambda1 = p.lambda1;
lambda2 = p.lambda2;
gamma = p.gamma;
features = p.t_features;
global_feat_params = p.t_global;
num_frames = numel(p.img_files);
s_frames = p.s_frames;
video_path = p.video_path;
% used for benchmark
rect_positions = zeros(num_frames, 4);
pos = p.init_pos;
target_sz = p.target_sz;
% Hann (cosine) window
hann_window_cosine = single(hann(p.cf_response_size(1)) * hann(p.cf_response_size(2))');

% gaussian-shaped desired response, centred in (1,1)
% bandwidth proportional to target size
output_sigma{1} = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor{1} / p.hog_cell_size;
output_sigma{2} = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor{2} / p.hog_cell_size;
y{1} = gaussianResponse(p.cf_response_size, output_sigma{1});
yf{1} = fft2(y{1});
y{2} = gaussianResponse(p.cf_response_size, output_sigma{2});
yf{2} = fft2(y{2});

model_x_f = cell(2,1);
model_w_f = cell(2,1);
z  = cell(2,1);
z_f = cell(2,1);
kz_f = cell(2,1);
x = cell(2,1);
x_f = cell(2,1);
k_f = cell(2,1);


learning_rate_pwp = p.learning_rate_pwp;
% patch of the target + padding
patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);
% initialize hist model
new_pwp_model = true;
[bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
new_pwp_model = false;

%% from DSST ******************************************
scale_factor = 1;
base_target_sz = target_sz;
scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
ss = (1:p.num_scales) - ceil(p.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(p.num_scales,2) == 0
    scale_window = single(hann(p.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(p.num_scales));
end
ss = 1:p.num_scales;
scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);
if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
    p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
end
scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
% find maximum and minimum scales
min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));

% initialization
prior_weights = cell(1,2);
sample_weights = cell(1,2);
% prior_weights = [];
% sample_weights = [];
latest_ind = cell(1,2);
% latest_ind = [];
sample_frame{1} = nan(p.nSamples);
sample_frame{2} = nan(p.nSamples);

quality{1} = 1i*zeros(p.nSamples, 1, 'single');
quality{2} = 1i*zeros(p.nSamples, 1, 'single');

samples_xf{1} = 1i*zeros(p.nSamples, prod(p.cf_response_size) * nDim{1},'single');
samples_xf{2} = 1i*zeros(p.nSamples, prod(p.cf_response_size) * nDim{2},'single');

samplesf{1} = 1i*zeros(p.nSamples,p.cf_response_size(1),p.cf_response_size(2),'single');
samplesf{2} = 1i*zeros(p.nSamples,p.cf_response_size(1),p.cf_response_size(2),'single');

debug = 0;

%%
t_imread = 0;
%% MAIN LOOP
tic;
for frame = 1:num_frames
    if frame == 244
        aaaa = 1;
    end
    if frame>1
        tic_imread = tic;
        % Load the image at the current frame
        im = imread([s_frames{frame}]);
        t_imread = t_imread + toc(tic_imread);
        
        %% TESTING step
        im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area); 
        % color histogram (mask)
        likelihood_map = getColourMap(im_patch_cf, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
        likelihood_map(isnan(likelihood_map)) = 0;
        likelihood_map = imResample(likelihood_map, p.cf_response_size);
        likelihood_map = (likelihood_map + min(likelihood_map(:)))/(max(likelihood_map(:)) + min(likelihood_map(:)));
        if (sum(likelihood_map(:))/prod(p.cf_response_size)<0.01)
            likelihood_map = 1;
        end
        likelihood_map = max(likelihood_map, 0.1);
        % apply color mask to sample(or hann_window)
        hann_window =  hann_window_cosine .* likelihood_map;
        for M = 1:2
            z{M} = bsxfun(@times, get_features(im_patch_cf, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window);
            z_f{M} = double(fft2(z{M}));
            switch p.kernel_type
                case 'gaussian'
                    kz_f{M} = gaussian_correlation(z_f{M}, model_x_f{M}, p.tran_sigma{M});
                case 'polynomial'
                    kz_f{M} = polynomial_correlation(z_f{M}, model_x_f{M}, p.polya{M}, p.polyb{M});
                case 'linear'
                    kz_f{M} = sum(z_f{M} .* conj(model_x_f{M}), 3) / numel(z_f{M});
            end
        end
        
        response_cf{1} = real(ifft2(model_w_f{1} .* kz_f{1}));
        response_cf{2} = real(ifft2(model_w_f{2} .* kz_f{2}));
        
        % Crop square search region (in feature pixels).
        response_cf{1} = cropFilterResponse(response_cf{1}, ...
            floor_odd(p.norm_delta_area / p.hog_cell_size));
        response_cf{2} = cropFilterResponse(response_cf{2}, ...
            floor_odd(p.norm_delta_area / p.hog_cell_size));
        
        if p.hog_cell_size > 1
            % Scale up to match center likelihood resolution.
            response_cf{1} = mexResize(response_cf{1}, p.norm_delta_area,'auto');
            response_cf{2} = mexResize(response_cf{2}, p.norm_delta_area,'auto');
        end
        
        p1 = adaptive_weight(response_cf{1});
        p2 = adaptive_weight(response_cf{2});
        response_cf_all = (p1 .* response_cf{1}./max(response_cf{1}(:))) + (p2 .* response_cf{2}./max(response_cf{2}(:)));
        response = response_cf_all;

        [row, col] = find(response == max(response(:)), 1);
        center = (1+p.norm_delta_area) / 2;
        pos = pos + ([row, col] - center) / area_resize_factor;
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %% SCALE SPACE SEARCH
        im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
        xsf = fft(im_patch_scale,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.scale_lambda) ));
        recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
        %set the scale
        scale_factor = scale_factor * scale_factors(recovered_scale);
        
        if scale_factor < min_scale_factor
            scale_factor = min_scale_factor;
        elseif scale_factor > max_scale_factor
            scale_factor = max_scale_factor;
        end
        % use new scale to update bboxes for target, filter, bg and fg models
        target_sz = round(base_target_sz * scale_factor);
        p.avg_dim = sum(target_sz)/2;
        bg_area = round(target_sz + p.avg_dim * p.padding);
        if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
        if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end
        
        bg_area = bg_area - mod(bg_area - target_sz, 2);
        fg_area = round(target_sz - p.avg_dim * p.inner_padding);
        fg_area = fg_area + mod(bg_area - fg_area, 2);
        % Compute the rectangle with (or close to) params.fixed_area and same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
       %% Visualization for Debug
        if p.visualization_dbg == 1
            if frame == 2
                figure(2);
                set(gcf,'unit','normalized','position',[0,0,1,1]);
                subplot(3,4,1);im_handle1 = imshow(im_patch_cf);title('Patch');
                subplot(3,4,2);im_handle2 = surf(response_cf{1}, 'FaceColor','interp','EdgeColor','none');title(feat_type{1});colormap('jet');
                subplot(3,4,3);im_handle3 = surf(response_cf{2}, 'FaceColor','interp','EdgeColor','none');title(feat_type{2});colormap('jet');
                subplot(3,4,4);im_handle4 = surf(response, 'FaceColor','interp','EdgeColor','none');title('Response');colormap('jet');
            else
                set(im_handle1, 'CData', im_patch_cf);
                set(im_handle2, 'zdata', response_cf{1});
                set(im_handle3, 'zdata', response_cf{2});
                set(im_handle4, 'zdata', response);
            end
            drawnow
        end
    end
    
    %% Update the prior weights
    [prior_weights{1}, replace_ind{1}] = update_prior_weights(prior_weights{1}, sample_weights{1}, latest_ind{1}, frame, p, p.learning_rate_cf{1});
    [prior_weights{2}, replace_ind{2}] = update_prior_weights(prior_weights{2}, sample_weights{2}, latest_ind{2}, frame, p, p.learning_rate_cf{2});
    latest_ind = replace_ind;
    sample_frame{1}(replace_ind{1}) = frame;
    sample_frame{2}(replace_ind{2}) = frame;
    
    % Initialize the weight for the new sample
    if frame == 1
        sample_weights{1} = prior_weights{1};
        sample_weights{2} = prior_weights{2};
    else
        % ensure that the new sample always get its current prior weight      
        new_sample_weight{1} = p.learning_rate_cf{1};
        sample_weights{1} = sample_weights{1} * (1 - new_sample_weight{1}) / (1 - sample_weights{1}(replace_ind{1}));
        sample_weights{1}(replace_ind{1}) = new_sample_weight{1};
        sample_weights{1} = sample_weights{1} / sum(sample_weights{1});
        
        new_sample_weight{2} = p.learning_rate_cf{2};
        sample_weights{2} = sample_weights{2} * (2 - new_sample_weight{2}) / (2 - sample_weights{2}(replace_ind{2}));
        sample_weights{2}(replace_ind{2}) = new_sample_weight{2};
        sample_weights{2} = sample_weights{2} / sum(sample_weights{2});
    end
    
    %% Train and Update Model
    im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    
    for M = 1:2
        x{M} = bsxfun(@times, get_features(im_patch_bg, features, global_feat_params, feat_type{M}, layerInd{M}), hann_window_cosine);
        x_f{M} = double(fft2(x{M}));
        switch p.kernel_type
            case 'gaussian'
                k_f{M} = gaussian_correlation(x_f{M}, x_f{M}, p.tran_sigma{M});
            case 'polynomial'
                k_f{M} = polynomial_correlation(x_f{M}, x_f{M}, p.polya{M}, p.polyb{M});
            case 'linear'
                k_f{M} = sum(x_f{M} .* conj(x_f{M}), 3) / numel(x_f{M});
        end
    end
    
    % Store new sample
    % For updating appearance model
    samples_xf{1}(replace_ind{1},:) = reshape(x_f{1},[prod(p.cf_response_size) * nDim{1}, 1]);
    samples_xf{2}(replace_ind{2},:) = reshape(x_f{2},[prod(p.cf_response_size) * nDim{2}, 1]);
    % For updating correlator model
    samplesf{1}(replace_ind{1},:,:) = permute(k_f{1},[3 1 2]);
    samplesf{2}(replace_ind{2},:,:) = permute(k_f{2},[3 1 2]);
    
    %Iteratively updates filter parameters and sample weights
    for acs_iter = 1 : p.num_acs_iter
        % Update filter
        A11 = permute(sum(bsxfun(@times, sample_weights{1}, bsxfun(@rdivide, bsxfun(@times, samplesf{1}, conj(samplesf{1})), samplesf{1}(replace_ind{1},:,:))), 1), [2 3 1]) + p.lambda1 + gamma * conj(k_f{1});
        A1 = permute(sum(bsxfun(@times, sample_weights{1}, bsxfun(@rdivide, samplesf{1}, samplesf{1}(replace_ind{1},:,:))), 1), [2 3 1]);
        
        A22 = permute(sum(bsxfun(@times, sample_weights{2}, bsxfun(@rdivide, bsxfun(@times, samplesf{2}, conj(samplesf{2})), samplesf{2}(replace_ind{2},:,:))), 1), [2 3 1]) + p.lambda2 + gamma * conj(k_f{2});
        A2 = permute(sum(bsxfun(@times, sample_weights{2}, bsxfun(@rdivide, samplesf{2}, samplesf{2}(replace_ind{2},:,:))), 1), [2 3 1]);
 
        w_f{1} = conj(bsxfun(@rdivide, A22 .* A1 .* conj(yf{1}) + gamma * A2 .* conj(yf{2}) .* conj(k_f{2}),...
                                   A11 .* A22 - gamma^2 * conj(k_f{2} .* k_f{1})));
        w_f{2} = conj(bsxfun(@rdivide, A11 .* A2 .* conj(yf{2}) + gamma * A1 .* conj(yf{1}) .* conj(k_f{1}),...
                                   A22 .* A11 - gamma^2 * conj(k_f{1} .* k_f{2})));                       

        % Update sample weights (alpha in the paper), and calculate loss
        if frame > p.sample_burnin 
            sample_loss = compute_loss(w_f, samplesf, yf, p);
            sample_weights{1} = update_weights(sample_loss{1}, prior_weights{1}, frame, p, p.sample_reg{1});
            sample_weights{2} = update_weights(sample_loss{2}, prior_weights{2}, frame, p, p.sample_reg{2});
        else
            sample_weights{1} = prior_weights{1};
            sample_weights{2} = prior_weights{2};
        end
        
    end
    quality{1}(:, replace_ind{1}) = sample_weights{1} .* prior_weights{1} / sum(sample_weights{1} .* prior_weights{1},1);
    quality{2}(:, replace_ind{2}) = sample_weights{2} .* prior_weights{2} / sum(sample_weights{1} .* prior_weights{2},1);

    model_x_f{1} = reshape(sum(bsxfun(@times, quality{1}(:, replace_ind{1}), samples_xf{1}), 1), [p.cf_response_size(1), p.cf_response_size(2), nDim{1}]);
    model_x_f{2} = reshape(sum(bsxfun(@times, quality{2}(:, replace_ind{2}), samples_xf{2}), 1), [p.cf_response_size(1), p.cf_response_size(2), nDim{2}]);

    model_w_f = w_f;
    
    if frame ~= 1
        % BG/FG MODEL UPDATE   patch of the target + padding
        im_patch_color = getSubwindow(im, pos, p.norm_bg_area, bg_area*(1-p.inner_padding));
        [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_color, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, learning_rate_pwp);
    end
    %% Upadate Scale
    im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
    xsf = fft(im_patch_scale,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
        sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
    end
    % update bbox position
    if (frame == 1)
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    end
    rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];
    rect_positions(frame,:) = rect_position;
    
    elapsed_time = toc;
    if p.fout > 0,  fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', rect_position(1),rect_position(2),rect_position(3),rect_position(4));   end
    %% Visualization
    fontSizeAxis = 12;
    if p.visualization_dbg == 1
        plot_frames = nan(num_frames,1); plot_sample = nan(num_frames,1); plot_prior = nan(num_frames,1);
        max_ind = min(frame, p.nSamples);
        [sorted_frames, ind] = sort(sample_frame{1}(1:max_ind));
        plot_frames(sorted_frames) = sorted_frames;
        plot_sample(sorted_frames) = sample_weights{1}(ind);
        plot_prior(sorted_frames) = prior_weights{1}(ind);
        % Sample weights
        if frame == 1
            figure(2);
            subplot(3,4,[7,8]); im_handle5 = plot(plot_frames, plot_prior, 'or-', 'linewidth',1.5, 'markersize', 3);
            hold on;
            subplot(3,4,[7,8]); im_handle6 = plot(plot_frames, plot_sample, 'xb-', 'linewidth',1.5, 'markersize', 3);
            hold off;
            title('Temporal-dependent factor and sample quality');
            legend({'Temporal-dependent factor', 'Sample quality'}, 'location', 'northwest');
            axis([1 num_frames 0 1.1*max([sample_weights{1}; prior_weights{1}])]);
%             set(gca, 'fontSize', fontSizeAxis);
        else
            set(im_handle5, 'xdata', plot_frames, 'ydata', plot_prior);
            set(im_handle6, 'xdata', plot_frames, 'ydata', plot_sample);
        end
        plot_frames = nan(num_frames,1); plot_sample = nan(num_frames,1); plot_prior = nan(num_frames,1);
        max_ind = min(frame, p.nSamples);
        [sorted_frames, ind] = sort(sample_frame{2}(1:max_ind));
        plot_frames(sorted_frames) = sorted_frames;
        plot_sample(sorted_frames) = sample_weights{2}(ind);
        plot_prior(sorted_frames) = prior_weights{2}(ind);
        
        if frame == 1
            figure(2);
            subplot(3,4,[11,12]); im_handle7 = plot(plot_frames, plot_prior, 'or-', 'linewidth',1.5, 'markersize', 3);
            hold on;
            subplot(3,4,[11,12]); im_handle8 = plot(plot_frames, plot_sample, 'xb-', 'linewidth',1.5, 'markersize', 3);
            title('Temporal-dependent factor and sample quality');
            xlabel('Frame number');
            legend({'Temporal-dependent factor', 'Sample quality'}, 'location', 'northwest')
            hold off;
            axis([1 num_frames 0 1.1*max([sample_weights{2}; prior_weights{2}])]);
%             set(gca, 'fontSize', fontSizeAxis);
        else
            set(im_handle7, 'xdata', plot_frames, 'ydata', plot_prior);
            set(im_handle8, 'xdata', plot_frames, 'ydata', plot_sample);
        end

        if frame == 1   %first frame, create GUI
            figure(2);
            subplot(3, 4, [5,6,9,10]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g', 'LineWidth',2);
            rect_handle2 = rectangle('Position',rect_position_padded, 'LineWidth',2, 'LineStyle','--', 'EdgeColor','b');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
            title('Result');
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im);
                set(rect_handle, 'Position', rect_position);
                set(rect_handle2, 'Position', rect_position_padded);
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end
        
        drawnow
    end
end

%% save data for benchmark
results.type = 'rect';
results.res = rect_positions;
results.fps = num_frames/(elapsed_time - t_imread);

end