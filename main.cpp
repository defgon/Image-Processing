#include <iostream>
#include "Convertor.h"
#include "Image.h"

int main() {
    Image a("smpls\\lady.ppm");

    a.fft2D(a.CMatrixPix, true);
    a.fftshift2D(a.CMatrixPix);
    a.Frequencyfilter(50,false);
    a.MakePicturesFromMatrix("dft\\1.ppm", "dft\\2.ppm", a.CMatrixPix);
    //a.fftshift2D(a.CMatrixPix);
    //a.fft2D(a.CMatrixPix, false);
    //a.ImageIFFT("smpls\\lady2.ppm",1);

    return 0;
}
