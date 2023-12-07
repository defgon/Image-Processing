#include <iostream>
#include "Convertor.h"
#include "Image.h"

int main() {
    Image a("smpls\\lady.ppm");

    a.fft2D(a.CMatrixPix, true);
    a.fftshift2D(a.CMatrixPix);
    a.Frequencyfilter();
    //a.makeMagnitude();
    //a.magnitudeSpec(10);
    //a.MakePicFromMagnitude("dft\\magSpec.ppm", a.MagnitudeSpec,10);
    //a.MakePicFromMagnitude("dft\\mag.ppm", a.Magnitude);
    a.fftshift2D(a.CMatrixPix);
    a.fft2D(a.CMatrixPix, false);
    a.ImageIFFT("smpls\\lady2.ppm",20);

    return 0;
}
