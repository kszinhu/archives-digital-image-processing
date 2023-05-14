% ---
% Course: Processamento de Imagens Digitais
% Student: Cassiano Henrique Aparecido Rodrigues
% Activity: Lista de Exercícios
% ---

clear all;
clf;
close all;

% Não consegui testar por não ter a imagem :c

% Usando a função built-in do Matlab para detectar círculos

function [x0, y0, r] = detect_circle(I)
  % Aplica a transformada de Hough para detectar círculos
  [centers, radii, ~] = imfindcircles(I, [10 100]);

  % Verifica se círculos foram encontrados
  if isempty(centers)
      error('Nenhum círculo foi encontrado na imagem.');
  end

  % Retorna as coordenadas do centro e o raio do círculo com maior diâmetro
  [~, idx] = max(radii);
  x0 = centers(idx, 1);
  y0 = centers(idx, 2);
  r = radii(idx);
end

% Usando na mão, a transformada de Hough para detectar círculos

function [x0, y0, r] = detect_circle_hand(I)
  bw = imbinarize(I);

  % Encontra os pontos de borda na imagem binária
  edge_points = edge(bw);

  % Define os parâmetros da transformada de Hough
  [rows, cols] = size(bw);
  max_radius = min(rows, cols) / 2;
  num_theta = 360;
  num_rho = round(2 * pi * max_radius);
  accum_matrix = zeros(num_rho, num_theta);

  % Realiza a transformada de Hough
  for x = 1:cols
      for y = 1:rows
          if edge_points(y, x)
              for theta = 0:359 % 0 a 360 graus
                  theta_rad = theta * pi / 180;
                  rho = round(x * cos(theta_rad) + y * sin(theta_rad));
                  accum_matrix(rho, theta+1) = accum_matrix(rho, theta+1) + 1;
              end
          end
      end
  end

  % Encontra os parâmetros da circunferência com maior voto na transformada de Hough
  [max_votes, max_index] = max(accum_matrix(:));
  [rho_index, theta_index] = ind2sub(size(accum_matrix), max_index);

  % Calcula as coordenadas do centro e o raio da circunferência
  x0 = round(cols / 2) + rho_index * cos((theta_index-1) * pi / 180);
  y0 = round(rows / 2) + rho_index * sin((theta_index-1) * pi / 180);
  r = rho_index;
end
