clear all;
clf;
close all;

img = imread('cameraman.tif');

[h, l] = size(img);

histogram = zeros(1, 256);

for i = 1:h
    for j = 1:l
        histogram(img(i, j)) = histogram(img(i, j)) + 1;
    end
end

plot(1:256, histogram, 'r');