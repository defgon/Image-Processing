#include <iostream>
#include "Convertor.h"
#include "Image.h"

int main() {
    Image a("smpls\\dog.ppm");

    a.fft2D(a.CMatrixPix, true);
    a.fftshift2D(a.CMatrixPix);
    //a.Frequencyfilter();
    a.MakePicturesFromMatrix("dft\\dogSFT.ppm", "dft\\dogSFT2.ppm", a.CMatrixPix);
    //a.makeMagnitude();
    //a.magnitudeSpec(10);
    //a.MakePicFromMagnitude("dft\\magSpec.ppm", a.MagnitudeSpec,10);
    //a.MakePicFromMagnitude("dft\\mag.ppm", a.Magnitude,1000);
    //a.fftshift2D(a.CMatrixPix);
    a.fft2D(a.CMatrixPix, false);
    a.ImageIFFT("smpls\\dog2.ppm");

    return 0;
}
