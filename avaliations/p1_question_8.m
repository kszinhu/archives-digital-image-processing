% ---
% Course: Processamento de Imagens Digitais
% Student: Cassiano Henrique Aparecido Rodrigues
% Activity: P1_Plus - Questão 8
% ---

clear all;
clf;
close all;

% Cria a imagem de entrada com 60 níveis de cinza
input_mono_img = uint8(randi([0 60], 256, 256));
output_mono_img = uint8(zeros(256, 256));

% Calcula a transformação linear para alterar a escala de níveis de cinza
for i = 1:256
    for j = 1:256
        % Y = (X - Xmin) * (Ymax - Ymin) / (Xmax - Xmin) + Ymin
        output_mono_img(i, j) = (input_mono_img(i, j) - 0) * (210 - 10) / (60 - 0) + 10;
    end
end

% Mostra a imagem de entrada
subplot(1, 2, 1);
imshow(input_mono_img);
title('Imagem de Entrada');

% Mostra a imagem de saída
subplot(1, 2, 2);
imshow(output_mono_img);
title('Imagem de Saída');
