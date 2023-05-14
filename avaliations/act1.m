% ---
% Course: Processamento de Imagens Digitais
% Student: Cassiano Henrique Aparecido Rodrigues
% Activity: Atividade Avaliativa 1
% ---

% TODO: Está com erros (arrumar mais tarde)

clear all;
clf;
close all;

number_of_quadrants = 4;
number_of_imgs = 112;

splited_imgs = {};
descriptor_array = {};
weight_matrix = [1 2 4; 128 0 8; 64 32 16];

% number of hits is the number of times that the descriptor of the image is one of numbers of quadrants of the same image (Object)
number_of_hits.euclidian = 0;
number_of_hits.chi_distance = 0;
number_of_misses.euclidian = 0;
number_of_misses.chi_distance = 0;

% read each image on assets assets/D{1...10}.gif
for i = 1:1:number_of_imgs
  current_img = imread(sprintf('assets/D%d.gif', i));
  [width, height] = size(current_img);
 
  % split the image in 4 quadrants
  splited_imgs{i, 1} = current_img(1:width/2, 1:height/2);
  splited_imgs{i, 2} = current_img(1:width/2, height/2+1:height);
  splited_imgs{i, 3} = current_img(width/2+1:width, 1:height/2);
  splited_imgs{i, 4} = current_img(width/2+1:width, height/2+1:height);
  for j = 1:1:number_of_quadrants
    descriptor_array{i, j} = linear_binary_pattern(splited_imgs{i, j}, weight_matrix(:,:));
  end
end

for i = 1:1:number_of_imgs % for each image
  euclidian_distance_array = zeros(1, number_of_imgs); 
  chi_distance_array = zeros(1, number_of_imgs);
  
  for q = 1:1:number_of_quadrants % for each quadrant calculate the distance between the descriptor of the quadrant and all others  
    for k = 1:1:number_of_imgs % for each image
      if (k == i) % if it's the same image, skip
        continue;
      end
      euclidian_distance_array(k) = euclidian_distance(descriptor_array{i, q}, descriptor_array{k, q});
      chi_distance_array(k) = chi_distance(descriptor_array{i, q}, descriptor_array{k, q});
    end
    [euclidian_min_distance, euclidian_index] = min(euclidian_distance_array);
    [chi_min_distance, chi_distance_index] = min(chi_distance_array);

    % display(fprintf('descriptor_array: %d', descriptor_array{i, q}));
    % display(fprintf('Imagem: %d, Quadrante: %d -> Euclidian: %d, Chi Distance: %d', i, q, euclidian_index, chi_distance_index));

    % se o euclidian_index has the same value of i on split_imgs, it's a hit (the descriptor is from the same image)
    if (euclidian_index == i)
      number_of_hits.euclidian = number_of_hits.euclidian + 1;
    else
      number_of_misses.euclidian = number_of_misses.euclidian + 1;
      display(fprintf('Imagem esperada: %d, Imagem encontrada: %d', i, euclidian_index));
    end

    if (chi_distance_index == i)
      number_of_hits.chi_distance = number_of_hits.chi_distance + 1;
    else
      number_of_misses.chi_distance = number_of_misses.chi_distance + 1;
      display(fprintf('Imagem esperada: %d, Imagem encontrada: %d', i, chi_distance_index));
    end
  end
end

display(fprintf('\n Taxa de acerto Euclidian: %d', hit_rate(number_of_hits.euclidian, number_of_misses.euclidian)));
display(fprintf('\n Taxa de acerto Chi Distance: %d', hit_rate(number_of_hits.chi_distance, number_of_misses.chi_distance)));

function [distance] = chi_distance(array_1, array_2)
  distance = sum((array_1 - array_2).^2) / sum(array_1 + array_2);
end

function [distance] = euclidian_distance(array_1, array_2)
  distance = sqrt(sum((array_1 - array_2).^2));
end

function [tax_hit] = hit_rate(number_of_hits, number_of_misses)
  value = number_of_hits / (number_of_hits + number_of_misses);
  tax_hit = strcat(num2str(value*100), '%');
end

function [descriptor_array] = linear_binary_pattern(img, weight_matrix)
  [width, height] = size(img);
  descriptor_array = zeros(1, 256); % amount of times the value appears in the histogram

  % lbp using weight matrix comparing center value with neighbors
  % split img on 3x3 matrix and get the center value
  for i = 2:1:width-1
    for j = 2:1:height-1
      center_value = img(i, j); 
      center_index = (i-2)*(height-2)+(j-1);
      neighbors = [img(i-1, j-1) img(i-1, j) img(i-1, j+1); img(i, j-1) center_value img(i, j+1); img(i+1, j-1) img(i+1, j) img(i+1, j+1)];

      % if neighbors value is greater than center value, set to 1, else set to 0 and multiply by weight matrix
      descriptor_array(center_index) = sum(sum((neighbors >= center_value) .* weight_matrix));

      if center_index == 256
          break;
      end
    end
    if center_index == 256
        break;
    end
  end
end
