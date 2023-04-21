clear all;
clf;
close all;

i_img = imread('cameraman.tif');

% 2 - Copy image I to J
j_img = i_img;

% 3 - Apply Salt and Pepper noise to image I with 5% noise (without built-in functions)
% j_img = imnoise(i_img, 'salt & pepper', 0.05); using built-in
j_img = salt_pepper_noise(i_img, 0.05);

% 4 - Apply Median and Mean filter
j_img_median = medfilt2(j_img);
j_img_mean = imfilter(j_img, fspecial('average', 3));

% 5 - Calculate ME, MAE, MSE, RMSE, NMSE, PSNR, COV, COR, JAC, SSIM metrics
% 6 - Create table "Métricas" with all metrics with 

noise_max = 0.5; % 50%
initial_noise = 0.05; % initial noise percentage
noise_increment = 0.05; % increment of noise percentage
num_steps = noise_max / noise_increment;
index = 1;

metrics_names = {'ME', 'MAE', 'MSE', 'RMSE', 'NMSE', 'PSNR', 'COV', 'COR', 'JAC', 'SSIM'};
metrics_row_names = {};
metrics_values = zeros(num_steps, length(metrics_names), 2);

% Apply metrics to matrics_values matrix
for current_noise = initial_noise:noise_increment:noise_max
    j_img = salt_pepper_noise(i_img, current_noise);

    i_imgs{index} = i_img;
    j_imgs{index} = j_img;

    % Apply Median and Mean filter
    j_imgs_median{index} = medfilt2(j_img);
    j_imgs_mean{index} = imfilter(j_img, fspecial('average', 3));

    % Calculate ME, MAE, MSE, RMSE, NMSE, PSNR, COV, COR, JAC, SSIM metrics    
    metrics_values(index, 1, 1) = maximum_error(i_img, j_imgs_median{index});
    metrics_values(index, 1, 1) = maximum_error(i_img, j_imgs_mean{index});
    metrics_values(index, 2, 1) = mean_error(i_img, j_imgs_median{index});
    metrics_values(index, 2, 2) = mean_error(i_img, j_imgs_mean{index});
    metrics_values(index, 3, 1) = quadratic_mean_error(i_img, j_imgs_median{index});
    metrics_values(index, 3, 2) = quadratic_mean_error(i_img, j_imgs_mean{index});
    metrics_values(index, 4, 1) = root_quadratic_mean_error(i_img, j_imgs_median{index});
    metrics_values(index, 4, 2) = root_quadratic_mean_error(i_img, j_imgs_mean{index});
    metrics_values(index, 5, 1) = normalized_quadratic_mean_error(i_img, j_imgs_median{index});
    metrics_values(index, 5, 2) = normalized_quadratic_mean_error(i_img, j_imgs_mean{index});
    metrics_values(index, 6, 1) = peak_signal_noise_ratio(i_img, j_imgs_median{index});
    metrics_values(index, 6, 2) = peak_signal_noise_ratio(i_img, j_imgs_mean{index});
    metrics_values(index, 7, 1) = covariance(i_img, j_imgs_median{index});
    metrics_values(index, 7, 2) = covariance(i_img, j_imgs_mean{index});
    metrics_values(index, 8, 1) = correlation(i_img, j_imgs_median{index});
    metrics_values(index, 8, 2) = correlation(i_img, j_imgs_mean{index});
    metrics_values(index, 9, 1) = jaccard(i_img, j_imgs_median{index});
    metrics_values(index, 9, 2) = jaccard(i_img, j_imgs_mean{index});
    metrics_values(index, 10, 1) = structured_similarity_index(i_img, j_imgs_median{index}, 0.03, 255);
    metrics_values(index, 10, 2) = structured_similarity_index(i_img, j_imgs_mean{index}, 0.03, 255);
    
    % Add current_noise to metrics_row_names and concatenate with "%"
    metrics_row_names{index} = strcat(num2str(current_noise * 100), '%');

    index = index + 1;
end

metric_table_median = array2table(metrics_values(:, :, 1), 'VariableNames', metrics_names, 'RowNames', metrics_row_names);
metric_table_mean = array2table(metrics_values(:, :, 2), 'VariableNames', metrics_names, 'RowNames', metrics_row_names);

% 9 - Display list of imagens (i, j, j_median, j_mean) respective with all noises
montage([i_imgs; j_imgs; j_imgs_median; j_imgs_mean], 'Size', [num_steps, 4]);

% 10 - Display table
disp('Métricas - Mediana');
disp(metric_table_median);
disp('Métricas - Média');
disp(metric_table_mean);

% 11 - Display on graph each metric with respect to the noise percentage
figure;
tiledlayout(length(metrics_names)/2, 2);

for i = 1:length(metrics_names)
    nexttile
    plot(metrics_values(:, i, 1));
    hold on;
    plot(metrics_values(:, i, 2));
    legend('Mediana', 'Média');
    title(metrics_names(i));
end

function [img] = salt_pepper_noise(img, noise_percent)
    [rows, cols] = size(img);
    num_pixels = rows * cols;
    num_noise_pixels = round(num_pixels * noise_percent);
    noise_pixels = randperm(num_pixels, num_noise_pixels);
    for i = 1:num_noise_pixels
        pixel = noise_pixels(i);
        row = max(floor(pixel / cols), 1);
        col = max(mod(pixel, cols), 1);
        img(row, col) = 255 * randi([0, 1]);
    end
end

% Erro Máximo (ME)
function [max_error] = maximum_error(img_f, img_g)
    [rows, cols] = size(img_f);
    max_error = 0;
    for i = 1:rows
        for j = 1:cols
            error = cast(abs(img_f(i, j) - img_g(i, j)), "uint32");
            if error > max_error
                max_error = error;
            end
        end
    end
end

% Erro Médio Absoluto (MAE)
function [mean_error] = mean_error(img_f, img_g)
    [rows, cols] = size(img_f);
    sum_error = 0;
    for i = 1:rows
        for j = 1:cols
            error = cast(abs(img_f(i, j) - img_g(i, j)), "uint32");
            sum_error = sum_error + error;
        end
    end
    mean_error = sum_error / (rows * cols);
end

% Erro Quadrático Médio (MSE)
function [quad_mean_error] = quadratic_mean_error(img_f, img_g)
    [rows, cols] = size(img_f);
    sum_error = 0;
    for i = 1:rows
        for j = 1:cols
            error = cast(abs(img_f(i, j) - img_g(i, j)), "uint32");
            sum_error = sum_error + error^2;
        end
    end
    quad_mean_error = sum_error / (rows * cols);
end

% Raiz do Erro Quadrático Médio (RMSE)
function [root_quad_mean_error] = root_quadratic_mean_error(img_f, img_g)
    [rows, cols] = size(img_f);
    sum_error = 0;
    for i = 1:rows
        for j = 1:cols
            error = cast(abs(img_f(i, j) - img_g(i, j)), "uint32");
            sum_error = sum_error + error^2;
        end
    end
    root_quad_mean_error = sqrt(cast(sum_error / (rows * cols), "double"));
end

% Erro Médio Quadrático Normalizado (NMSE)
function [norm_quad_mean_error] = normalized_quadratic_mean_error(img_f, img_g)
    [rows, cols] = size(img_f);
    sum_error = 0;
    sum_img1 = 0;
    for i = 1:rows
        for j = 1:cols
            error = cast(img_f(i, j) - img_g(i, j), "double");
            sum_error = sum_error + error^2;
            sum_img1 = sum_img1 + cast(img_f(i, j), "double")^2;
        end
    end
    norm_quad_mean_error = sum_error / sum_img1;
end

% Relação sinal-ruído de pico (PSNR)
function [peak_signal_noise_ratio] = peak_signal_noise_ratio(img_f, img_g)
    MNL_max = 255; % para imagens em tons de cinza 8 bits de profundidade
    [rows, cols] = size(img_f);
    sum_error = 0;
    for i = 1:rows
        for j = 1:cols
            error = abs(cast(img_f(i, j) - img_g(i, j), "double"));
            sum_error = sum_error + error^2;
        end
    end
    peak_signal_noise_ratio = 10 * log10(cast(MNL_max^2 / (sum_error / (rows * cols)), "double"));
end

% Covariância (COV)
function [covariance] = covariance(img_f, img_g)
    mi_f = mean(img_f, 'all'); % nível médio de cinza da imagem f
    mi_g = mean(img_g, 'all'); % nível médio de cinza da imagem g

    [rows, cols] = size(img_f);
    sum_error = 0;
    for i = 1:rows
        for j = 1:cols
            error = cast((img_f(i, j) - mi_f) * (img_g(i, j) - mi_g), "double");
            sum_error = sum_error + error;
        end
    end
    covariance = sum_error / (rows * cols);
end

% Correlação (COR)
function [correlation] = correlation(img_f, img_g)
    mi_f = mean(img_f, 'all'); % nível médio de cinza da imagem f
    mi_g = mean(img_g, 'all'); % nível médio de cinza da imagem g

    [rows, cols] = size(img_f);
    sum_error = 0;
    sum_error_f = 0;
    sum_error_g = 0;
    for i = 1:rows
        for j = 1:cols
            error = cast((img_f(i, j) - mi_f) * (img_g(i, j) - mi_g), "double");
            sum_error = sum_error + error;
            sum_error_f = cast(sum_error_f + (img_f(i, j) - mi_f)^2, "double");
            sum_error_g = cast(sum_error_g + (img_g(i, j) - mi_g)^2, "double");
        end
    end
    correlation = cast(sum_error / sqrt(sum_error_f * sum_error_g), "double");
end

% Coeficiente de Jaccard (JAC)
function [jaccard_value] = jaccard(img_f, img_g)
    [rows, cols] = size(img_f);
    identity_cof = 0;
    for i = 1:rows
        for j = 1:cols
            if img_f(i, j) == img_g(i, j)
                identity_cof = identity_cof + 1;
            end
        end
    end
    jaccard_value = cast(identity_cof / (rows * cols), "double");
end

% Structured Similarity Index (SSIM)
function [ssim_value] = structured_similarity_index(img_f, img_g, K, L)
    C = (K * L)^2;

    % Divida c/ imagem em blocos 8x8
    [rows, cols] = size(img_f);
    blocks_f = zeros(rows/8, cols/8);
    blocks_g = zeros(rows/8, cols/8);
    
    for i = 1:rows/8
        for j = 1:cols/8
            % Calcule a média e o desvio padrão de cada bloco
            % Calcule a covariância entre os blocos
            mean_f = mean(img_f(i*8-7:i*8, j*8-7:j*8), 'all');
            mean_g = mean(img_g(i*8-7:i*8, j*8-7:j*8), 'all');

            std_f = std(cast(img_f(i*8-7:i*8, j*8-7:j*8), "double"), 0, 'all');
            std_g = std(cast(img_g(i*8-7:i*8, j*8-7:j*8), "double"), 0, 'all');

            cov_f_g = covariance(img_f(i*8-7:i*8, j*8-7:j*8), img_g(i*8-7:i*8, j*8-7:j*8));

            % Calcule o SSIM
            ssim_values{i, j} = (2 * mean_f * mean_g + C) * (2 * cov_f_g + C) / (mean_f^2 + mean_g^2 + C) / (std_f^2 + std_g^2 + C);
        end
    end
    % Calcule a média dos SSIMs
    ssim_value = mean(cell2mat(ssim_values), 'all');
end