% ---
% Course: Processamento de Imagens Digitais
% Student: Cassiano Henrique Aparecido Rodrigues
% Activity: P1_Plus - Questão 10
% ---


%{
    Questão 8: No processamento de textura, um descritor importante, 
    denominado energia, pode ser obtido a partir das matrizes 
    de co-ocorrências de tons de cinza (GLCM). Dadas as duas GLCM 
    abaixo, mostre quais imagens poderiam gerar estas matrizes
    e explique para qual delas o valor da energia será maior.
%}

clear all;
clf;
close all;

% Carrega GLCMs, obtidas na questão 10
glcm_1 = [5/35 0 0 0 0 0;
         0 5/35 0 0 0 0;
         0 0 5/35 0 0 0;
         0 0 0 5/35 0 0;
         0 0 0 0 5/35 0;
         0 0 0 0 0 5/35];

glcm_2 = [0 0 0 0 0 0;
         0 0 0 0 0 0;
         0 0 35/35 0 0 0;
         0 0 0 0 0 0;
         0 0 0 0 0 0;
         0 0 0 0 0 0];

% Calcula a Energia Sum_i Sum_j (p(i,j))^2
energy_1 = 0;
energy_2 = 0;
for i = 1:6
    for j = 1:6
        energy_1 = energy_1 + glcm_1(i,j)^2;
        energy_2 = energy_2 + glcm_2(i,j)^2;
    end
end

fprintf("Energia da GLCM 1: %f\n", energy_1);
fprintf("Energia da GLCM 2: %f\n", energy_2);

%{
    A GLCM 2 é a matriz de co-ocorrência de uma imagem com textura mais homogênea,
    pois possui apenas um valor diferente de zero, na posição (3,3). Portanto, a 
    energia da GLCM 2 é maior que a energia da GLCM 1, que possui 5 valores 
    diferentes de zero. 
%}

img_to_glcm2 = [3 3 3 3 3 3 3;
                3 3 3 3 3 3 3;
                3 3 3 3 3 3 3;
                3 3 3 3 3 3 3;
                3 3 3 3 3 3 3;
                3 3 3 3 3 3 3;
                3 3 3 3 3 3 3];

img_to_glcm1 = [0 0 0 0 0 0 0
                1 1 1 1 1 1 1
                2 2 2 2 2 2 2
                3 3 3 3 3 3 3
                4 4 4 4 4 4 4
                5 5 5 5 5 5 5
                6 6 6 6 6 6 6];


% check glcm of source image
glcm_check_1 = graycomatrix(img_to_glcm1, 'NumLevels', 6, 'GrayLimits', [0 5], 'Offset', [0 1]);
glcm_check_2 = graycomatrix(img_to_glcm2, 'NumLevels', 6, 'GrayLimits', [0 5], 'Offset', [0 1]);

figure(1);
subplot(1,4,1);
imshow(img_to_glcm1);
title('Imagem 1');

subplot(1,4,2);
imshow(img_to_glcm2);
title('Imagem 2');

subplot(1,4,3);
imshow(glcm_check_1);
title('GLCM 1');

subplot(1,4,4);
imshow(glcm_check_2);
title('GLCM 2');

