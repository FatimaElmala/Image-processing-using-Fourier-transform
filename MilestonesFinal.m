clc;
clear;
close all;
%FIRST MILESTONE:
%apply sin fuction
Fs = 5000; %sampling frequency
Ts = 1/Fs;
t = 0:Ts:1;
%testing with different frequencies
f1 = 50;
f2 = 100;
f3 = 400;
%sine fuctions with different frequencies 
y1 = sin(2*pi*f1*t);
y2 = sin(2*pi*f2*t);
y3 = sin(2*pi*f3*t);
sum_fun  = y1+y2+y3;
%plotting the sine waves in the time domain 
figure(1)
subplot(4,1,1);plot(t,y1);title("sine wave with frequency = 50");
subplot(4,1,2);plot(t,y2);title("sine wave with frequency = 100");
subplot(4,1,3); plot(t,y3);title("sine wave with frequency = 400");
subplot(4,1,4);plot(t,sum_fun);title("sum of the sine functions");
%fourier transform of the sine functions
f = -Fs/2:Fs/2;
Y1 = fftshift(fft(y1));
Y2 = fftshift(fft(y2));
Y3 = fftshift(fft(y3));
Y_SUM = fftshift(fft(sum_fun));
%plotting the fourier transform of the sine functions
figure(2)
subplot(4,1,1);plot(f,abs(Y1));title("fourier transform of sine with frequency = 50");
subplot(4,1,2);plot(f,abs(Y2));title("fourier transform of sine with frequency = 100");
subplot(4,1,3);plot(f,abs(Y3));title("fourier transform of sine with frequency = 400");
subplot(4,1,4);plot(f,abs(Y_SUM));title("fourier transform of the the sum function");


%import image
RGB = imread('D:\uni\year 3\semster 1\intro to communication\Project\egypt.jpg');
figure(3);imshow(RGB); title('original image(Egypt)');
%Transform a colored image to grey
I = rgb2gray(RGB);
figure(4); imshow(I); title('grey image(Egypt)');
%Adjust the contrast of the image
J = imadjust(I);
figure(5);imshow(I,[]);title('Contrast-adujsted Image (Egypt)');

%% ---------------------------------------------------------------------------------
%SECOND MILESTONE
%audio file to fourier transform

%reading sound wave 
[y,Fs] = audioread('D:\uni\year 3\semster 1\intro to communication\Project\clock.wav');
%playing the sound 
sound(y,Fs);
%plotting the sound wave befor the fourier transform
figure(6);plot(y); title(' sound wave');
%applying fourier transform to the sound wave 
data_fft = fft(y);
%shifting the frequency 
fft_shift= fftshift(data_fft);
freq=-length(fft_shift)/2:length(fft_shift)/2-1;
%displaying the fourier transform of the sound wave
figure(7);plot(freq,abs(fft_shift)); title('fourier transform of the sound wave');

%Get Fourier Transform of an image
F = fft2(I);
% Absolute of fourier transform of an image
S = abs(F);
figure(8);imshow(S,[]);title('Fourier transform of an image');
%Get the centered spectrum
Fsh = fftshift(F);
figure(9);imshow(abs(Fsh),[]);title('Centered fourier transform of Image');
%apply log transform
S2 = log(1+abs(Fsh));
figure(10);imshow(S2,[]);title('log transformed Image');
%reconstruct the Image
F = ifftshift(Fsh);
f = ifft2(F);
figure(11);imshow(f,[]),title('reconstructed Image');
%% --------------------------------------------------------------------------------------------------
%THIRD MILESTONE

%difference between the images in the spacial domain 
x=imread('D:\uni\year 3\semster 1\intro to communication\Project\sushi1.jpg');
y=imread('D:\uni\year 3\semster 1\intro to communication\Project\sushi2.jpg');

%resizing the images to make sure they are the same size to be able to
%perform the difference 
g=size(x);
y=imresize(y,[g(1),g(2)]);

figure(12);imshow(x);title('First image'); %displaying the first image 
figure(13);imshow(y);title('Second image'); %displaying the second image 
figure(14);imshow(x-y);title('Difference of two images'); %displaying the difference between the images


%---------------------------------------------------------------------------------------
a = imread('D:\uni\year 3\semster 1\intro to communication\Project\sushi1.jpg');          % reading image 1 
b = imread('D:\uni\year 3\semster 1\intro to communication\Project\sushi2.jpg');              %reading image 2 

% Convert to grayscale incase it is color
if size(size(a),2) == 3
a = rgb2gray(a);
end

a = im2double(a);                                   %convert the range of colors from 0-255 to 0-1

% Convert to grayscale incase it is color
if size(size(b),2) == 3
b = rgb2gray(b);
end

b = im2double(b);

figure(15)
subplot(4,2,1);imshow(a);title('Image 1');
subplot(4,2,2);imshow(b);title('Image 2');

[m,n] =size(a);           %Get size of image a
[x,y] = size(b);          %get size of image b  


disp(['The size of the image is ',num2str(m),' x ',num2str(n)])
maxallowed = min(m,n)/2;
do = input(['Enter the cutoff frequency (A number less than ',num2str(maxallowed),') :']);

disp(['The size of the image is ',num2str(x),' x ',num2str(y)])
maxallowed = min(x,y)/2;
do = input(['Enter the cutoff frequency (A number less than ',num2str(maxallowed),') :']);
  
A = fft2(a);              %fourier transform of image a
A_1 = fft2(b);            %fourier transform of image b



A1 = fftshift(A);            %shifting fourier tansform of image 1
A11 = abs(A1);               %getting the magnitude of the fourier transform of image 1

A_11 = fftshift(A_1);        %shifting fourier tansform of image 1
A_111 = abs(A11);            %getting the magnitude of the fourier transform of image 2

figure(15)
subplot(4,2,3);imshow(uint8(abs(A)));title('fourier transform of Image 1');
subplot(4,2,4);imshow(uint8(abs(A_1)));title('fourier transform of Image 2');


%arrays needed for the low pass filter for the first image 
Alow = zeros(m,n);
Ahigh = zeros(m,n);
d = zeros(m,n);
%low pass filter for first image 
for i=1:m
    for j=1:n
        d(i,j)=sqrt((i-(m/2))^2+(j-(n/2))^2);
        if d(i,j)<=do
            Alow(i,j)=A1(i,j);
            Ahigh(i,j)=0;
            filt(i,j) = 0;
        else
            Alow(i,j)=0;
            Ahigh(i,j)=A1(i,j);
            filt(i,j) = 1;
           
        end
    end
end

%arrays needed for the low pass filter 
Blow = zeros(x,y);
Bhigh = zeros(x,y);
e = zeros(x,y);
%low pass filter for first image 
for i=1:x
    for j=1:y
        e(i,j)=sqrt((i-(x/2))^2+(j-(y/2))^2);
        if d(i,j)<=do
            Blow(i,j)=A_11(i,j);
            Bhigh(i,j)=0;
            filt(i,j) = 0;
        else
            Blow(i,j)=0;
            Bhigh(i,j)=A_11(i,j);
            filt(i,j) = 1;
           
        end
    end
end




B = fftshift(Alow);                             %Reshifting the origin of filtered image
B1 = ifft2(B);                                  %Taking inverse fourier transform
B2 = abs(B1);                                   %Taking magnitude. (Low pass Filtered output image)
subplot(4,2,5);imshow(B2);title('low pass image1'); %display image 1 after the low pass filter 

C = fftshift(Blow);
C1 = ifft2(C);                                 
C2 = abs(C1);
subplot(4,2,6);imshow(B2);title('low pass image2'); %displaying image 2 after the low pass filter 

F1 = fft2(B2);                                   % fourier transform of the low pass filter of image 1 
figure(15)
subplot(4,2,7);imshow(uint8(abs(F1)));title("ft for lp Image1"); %displaying fourier transform of the low pass filter of image 1 
F2 = fft2(C2);                                   % fourier transform of the low pass filter of image 2
figure(15)
subplot(4,2,8);imshow(uint8(abs(F2)));title("ft for lp Image2");%displaying fourier transform of the low pass filter of image 2 


X=F1;
Y=F2;
g=size(X);
Y=imresize(Y,[g(1),g(2)]);
figure(16);imshow(X);title('First image');
figure(17);imshow(Y);title('Second image');
figure(18);imshow(X-Y);title('Difference of two images');


S = ifftshift(X-Y);            %inverse shift of the difference 
Sh = ifft2(S);                 %inverse fourier transform of the difference 
figure(19);imshow(Sh,[]);title('reconstructed Image'); %diplaying the difference of the two images 
