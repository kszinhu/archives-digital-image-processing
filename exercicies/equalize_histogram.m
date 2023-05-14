% ---
% Course: Processamento de Imagens Digitais
% Student: Cassiano Henrique Aparecido Rodrigues
% Activity: Lista de Exercícios
% ---

clear all;
clf;
close all;

img_a = [0 0 0 0 0 0 0 0 0 0; 0 1 1 1 1 1 2 2 2 0; 0 1 1 1 1 1 2 2 2 0; 0 2 2 2 2 2 2 2 0 0; 0 2 2 2 2 2 2 2 0 0; 0 0 0 0 0 0 0 0 0 0; 0 0 0 9 9 8 8 8 0 0; 0 0 0 9 9 8 8 9 0 0; 0 0 0 9 9 9 9 9 0 0; 0 0 0 0 0 0 0 0 0 0];

% 1. Calcular o histograma
histograma = histcounts(img_a, 10);

% 2. Calcular a função de distribuição acumulada (CDF)
cdf = cumsum(histograma);

% 3. Normalizar a CDF
cdf = cdf / numel(img_a);

% 4. Calcular os novos níveis de intensidade
novos_niveis = round(cdf * 9);

T = zeros(1, 10);
for i = 1:10
    T(i) = novos_niveis(i);
end

% 6. Aplicar a função de transformação T na imagem a
[n, m] = size(img_a);
img_equalizada = zeros(n, m);
for i = 1:n
    for j = 1:m
        img_equalizada(i, j) = T(img_a(i, j) + 1);
    end
end

% 7. Calcular o histograma da imagem equalizada
histograma_equalizado = histcounts(img_equalizada, 10);

% 8. Mostrar os resultados
figure;
subplot(2, 2, 1);
imshow(img_a, []);
title('Imagem original');
subplot(2, 2, 2);
bar(histograma);
title('Histograma da imagem original');
subplot(2, 2, 3);
imshow(img_equalizada, []);
title('Imagem equalizada');
subplot(2, 2, 4);
bar(histograma_equalizado);
title('Histograma da imagem equalizada');
  
% 9. Mostrar a função de transformação T
figure;
plot(T);
title('Função de transformação T');
xlabel('Níveis de intensidade originais');
ylabel('Novos níveis de intensidade');
