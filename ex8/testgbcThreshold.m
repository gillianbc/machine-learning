clc; clear; close all;
[epsilon F1] = selectThreshold([1 0 0 1 1]', [0.1 0.2 0.3 0.4 0.5]')



fprintf('Expected: epsilon =  0.40040 F1 =  0.57143\n');