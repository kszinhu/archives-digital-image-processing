% ---
% Course: Processamento de Imagens Digitais
% Student: Cassiano Henrique Aparecido Rodrigues
% Activity: Lista de Exercícios
% ---

clear all;
clf;
close all;

% img = imread('../assets/cameraman.tif');
img_a = [0 0 0 0 0 0 0 0 0 0; 0 1 1 1 1 1 2 2 2 0; 0 1 1 1 1 1 2 2 2 0; 0 2 2 2 2 2 2 2 0 0; 0 2 2 2 2 2 2 2 0 0; 0 0 0 0 0 0 0 0 0 0; 0 0 0 9 9 8 8 8 0 0; 0 0 0 9 9 8 8 9 0 0; 0 0 0 9 9 9 9 9 0 0; 0 0 0 0 0 0 0 0 0 0];

[h, l] = size(img_a);

% show img_a with intensity values 0-9
figure(2);
imshow(img_a, [0 9]);
title('img_a');

histogram = zeros(1, 10);

for i = 1:1:h
    for j = 1:1:l
        histogram(img_a(i, j) + 1) = histogram(img_a(i, j) + 1) + 1;
    end
end

% plot histogram, but change the x axis to 0-9 instead of 1-10
figure(1);
bar(0:9, histogram);
title('Histogram of img_a');
xlabel('Intensity');
ylabel('Number of pixels');
