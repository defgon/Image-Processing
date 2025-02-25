#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include "matrix.h"
#include <vector>
#include <ccomplex>
#include <limits>



const double PI = 3.14159265358979323846;
typedef std::complex<double> Complex;

struct Pixel {
    int r, g, b;
};

struct PixelD {
    double r, g, b;

    // operator *
    PixelD operator*(double scalar) const {
        PixelD result;
        result.r = r * scalar;
        result.g = g * scalar;
        result.b = b * scalar;
        return result;
    }
};

struct PixelC {
    Complex r, g, b;

    //operator =
    PixelC& operator=(const PixelC& other) {
        r = other.r;
        g = other.g;
        b = other.b;
        return *this;
    }

    //operator +
    PixelC operator+(const PixelC& other) const {
        PixelC result;
        result.r = r + other.r;
        result.g = g + other.g;
        result.b = b + other.b;
        return result;
    }

    //operator -
    PixelC operator-(const PixelC& other) const {
        PixelC result;
        result.r = r - other.r;
        result.g = g - other.g;
        result.b = b - other.b;
        return result;
    }

    //operator *
    PixelC operator*(const Complex& scalar) const {
        PixelC result;
        result.r = r * scalar;
        result.g = g * scalar;
        result.b = b * scalar;
        return result;
    }

    //operator /=
    PixelC& operator/=(int divisor) {
        r /= divisor;
        g /= divisor;
        b /= divisor;
        return *this;
    }

};

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sinc(double x) {
    if (x == 0.0) {
        return 1.0; // Defined value at x = 0
    }
    else {
        return std::sin(PI * x) / (PI * x); // Normalized sinc function
    }
}

struct Image {
    //attributes of Image
    Matrix<Pixel> MatrixPix;
    Matrix<double> MatrixGray;
    Matrix<PixelD> Magnitude;
    Matrix<PixelD> MagnitudeSpec;
    Matrix<PixelC> CMatrixPix;
    std::string type, width, height, RGB;
    std::string ImageName;

    // constructor
    Image(std::string Image) : ImageName(Image) {
        //basic matrix initializations
        ImgToMatrix();
        toComplex();
        ToGrayScale();
    }

    void ImgToMatrix() {
        std::ifstream image;
        image.open(ImageName);
        image >> type >> width >> height >> RGB;
        MatrixPix.resize(std::stoi(height), std::stoi(width));

        int r, g, b;
        for (int i = 0; i < MatrixPix.getRows(); i++) {
            for (int j = 0; j < MatrixPix.getColumns(); j++) {
                Pixel p;
                image >> r >> g >> b;
                p.r = r;
                p.g = g;
                p.b = b;
                MatrixPix(i, j) = p;
            }
        }
        image.close();
    }

    void toComplex() {
        CMatrixPix.resize(std::stoi(height), std::stoi(width));

        for (int i = 0; i < MatrixPix.getRows(); ++i) {
            for (int j = 0; j < MatrixPix.getColumns(); ++j) {
                CMatrixPix(i, j).r = Complex(static_cast<double>(MatrixPix(i, j).r), 0.0);
                CMatrixPix(i, j).g = Complex(static_cast<double>(MatrixPix(i, j).g), 0.0);
                CMatrixPix(i, j).b = Complex(static_cast<double>(MatrixPix(i, j).b), 0.0);
            }
        }
    }

    void fft2D(Matrix<PixelC>& data, bool dir) {

        if (dir) {
            data.resizeToNearestPowerOf2();
        }
        // 1D FFT for rows
        for (int i = 0; i < data.getRows(); ++i) {
            fft1D(data, dir, i);
        }
        // Transpon the matrix
        for (int i = 0; i < data.getRows(); ++i) {
            for (int j = i + 1; j < data.getColumns(); ++j) {
                std::swap(data(i, j), data(j, i));
            }
        }
        // 1D FFT for columns
        for (int i = 0; i < data.getRows(); ++i) {
            fft1D(data, dir, i);
        }
        // Transpon back to original
        for (int i = 0; i < data.getRows(); ++i) {
            for (int j = i + 1; j < data.getColumns(); ++j) {
                std::swap(data(i, j), data(j, i));
            }
        }
        // Resize back to the original size
        if (!dir) {
            data.rescaleToOriginalSize(std::stoi(height), std::stoi(width));
        }
    }

    void fft1D(Matrix<PixelC>& data, bool dir, int row) {
        int cols = data.getColumns();

        for (int i = 0; i < cols; i++) {
            int j = 0;
            for (int k = 0; (1 << k) < cols; ++k) {
                j |= ((i >> k) & 1) << (int(log2(cols)) - 1 - k);
            }
            if (j > i) {
                std::swap(data(row, i), data(row, j));
            }
        }

        for (int l = 2; l <= cols; l <<= 1) {
            double ang = 2 * PI / l * (dir ? -1 : 1);
            std::complex<double> wlen(std::cos(ang), std::sin(ang));
            for (int i = 0; i < cols; i += l) {
                std::complex<double> w(1);
                for (int j = 0; j < l / 2; ++j) {
                    auto u = data(row, i + j);
                    auto v = data(row, i + j + l / 2) * w;
                    data(row, i + j) = u + v;
                    data(row, i + j + l / 2) = u - v;
                    w *= wlen;
                }
            }
        }
        if (!dir) {
            for (int i = 0; i < cols; ++i) {
                data(row, i) /= cols;
            }
        }
    }

    void fftshift2D(Matrix<PixelC>& data) {
        // Apply 1D FFT on rows
        for (int i = 0; i < data.getRows(); ++i) {
            fftshift(data, i);
        }

        // Transpose the matrix
        for (int i = 0; i < data.getRows(); ++i) {
            for (int j = i + 1; j < data.getColumns(); ++j) {
                std::swap(data(i, j), data(j, i));
            }
        }

        // Apply 1D FFT on columns
        for (int i = 0; i < data.getRows(); ++i) {
            fftshift(data, i);
        }

        // Transpose back to the original orientation
        for (int i = 0; i < data.getRows(); ++i) {
            for (int j = i + 1; j < data.getColumns(); ++j) {
                std::swap(data(i, j), data(j, i));
            }
        }
    }

    void fftshift(Matrix<PixelC>& data, int row) {
        int N = data.getColumns();
        int midpoint = N / 2;

        std::vector<PixelC> temp(midpoint);

        for (int i = 0; i < midpoint; ++i) {
            temp[i] = data(row, i);
            data(row, i) = data(row, i + midpoint);
            data(row, i + midpoint) = temp[i];
        }
    }

    void Frequencyfilter(int radius,bool leqOrgeq) {
        int centerX = CMatrixPix.getColumns() / 2;
        int centerY = CMatrixPix.getRows() / 2;
        int radiusSquared = radius * radius;
        
        for (int i = 0; i < CMatrixPix.getRows(); ++i) {
            for (int j = 0; j < CMatrixPix.getColumns(); ++j) {
                int distanceSquared = (j - centerX) * (j - centerX) + (i - centerY) * (i - centerY);
                //filter inner or outer circle from middle
                if (leqOrgeq ? (distanceSquared < radiusSquared): (distanceSquared > radiusSquared)) {
                    CMatrixPix(i, j).r = 0;
                    CMatrixPix(i, j).g = 0;
                    CMatrixPix(i, j).b = 0;
                }
            }
        }
    }

    /*
    void makeMagnitude() {
        Magnitude.resize(CMatrixPix.getRows(), CMatrixPix.getColumns());
        for (int i = 0; i < CMatrixPix.getRows(); i++) {
            for (int j = 0; j < CMatrixPix.getColumns(); j++) {
                Magnitude(i, j).r = CMatrixPix(i, j).r.real() * CMatrixPix(i, j).r.real() + CMatrixPix(i, j).r.imag() * CMatrixPix(i, j).r.imag();
                Magnitude(i, j).r = sqrt(Magnitude(i, j).r);

                Magnitude(i, j).g = CMatrixPix(i, j).g.real() * CMatrixPix(i, j).g.real() + CMatrixPix(i, j).g.imag() * CMatrixPix(i, j).g.imag();
                Magnitude(i, j).g = sqrt(Magnitude(i, j).g);

                Magnitude(i, j).b = CMatrixPix(i, j).b.real() * CMatrixPix(i, j).b.real() + CMatrixPix(i, j).b.imag() * CMatrixPix(i, j).b.imag();
                Magnitude(i, j).b = sqrt(Magnitude(i, j).b);
            }
        }
    }

    void magnitudeSpec(int factor) {
        MagnitudeSpec.resize(Magnitude.getRows(), Magnitude.getColumns());
        for (int i = 0; i < Magnitude.getRows(); i++) {
            for (int j = 0; j < Magnitude.getColumns(); j++) {
                MagnitudeSpec(i, j).r = factor * log(Magnitude(i, j).r + 1);
                MagnitudeSpec(i, j).g = factor * log(Magnitude(i, j).g + 1);
                MagnitudeSpec(i, j).b = factor * log(Magnitude(i, j).b + 1);
            }
        }
    }

    void radialMask() {
        double center_x = Magnitude.getRows() / 2; // Center coordinates
        double center_y = Magnitude.getColumns() / 2;

        for (int y = 0; y < Magnitude.getRows(); ++y) {
            for (int x = 0; x < Magnitude.getColumns(); ++x) {
                double distance = sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y));
                double max_distance = sqrt(center_x * center_x + center_y * center_y);
                double value = 1 - (distance / max_distance);
                Magnitude(y, x) = Magnitude(y, x) * value;
            }
        }
    }

    void MakePicFromMagnitude(std::string ImageNameNew, Matrix<PixelD>& matrix, int konst) {
        std::ofstream newimage;
        newimage.open(ImageNameNew);

        newimage << type << std::endl;
        newimage << width << " " << height << std::endl;
        newimage << RGB << std::endl;
        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {
                int r, g, b;
                r = (int)std::round(std::abs(matrix(i, j).r)/konst);
                g = (int)std::round(std::abs(matrix(i, j).g) / konst);
                b = (int)std::round(std::abs(matrix(i, j).b) / konst);
                if (r > 255) { r = 255; }
                if (g > 255) { g = 255; }
                if (b > 255) { b = 255; }

                newimage << r << " " << g << " " << b << std::endl;
            }
        }
        newimage.close();
    }
    */
    void ImageIFFT(std::string ImageNameNew,double konst) {
           
        std::ofstream newimage1;
        newimage1.open(ImageNameNew);

        newimage1 << type << std::endl;
        newimage1 << width << " " << height << std::endl;
        newimage1 << RGB << std::endl;

        for (int i = 0; i < CMatrixPix.getRows(); i++) {
            for (int j = 0; j < CMatrixPix.getColumns(); j++) {
                int r, g, b;
                r = static_cast<int>(CMatrixPix(i, j).r.real() / konst);
                g = static_cast<int>(CMatrixPix(i, j).g.real() / konst);
                b = static_cast<int>(CMatrixPix(i, j).b.real() / konst);
                if (r > 255) { r = 255; }
                if (g > 255) { g = 255; }
                if (b > 255) { b = 255; }
                if (r < 0) { r = 0; }
                if (g < 0) { g = 0; }
                if (b < 0) { b = 0; }

                newimage1 << r << " " << g << " " << b << std::endl;

            }
        }
        newimage1.close();
        
    }

    /*
    void MakePicturesFromMatrix(std::string ImageNameNew1, std::string ImageNameNew2,Matrix<PixelC> matrix)
    {
        
        int konst = 100;
        std::ofstream newimage1;
        std::ofstream newimage2;
        newimage1.open(ImageNameNew1);
        newimage2.open(ImageNameNew2);

        newimage1 << type << std::endl;
        newimage1 << width << " " << height << std::endl;
        newimage1 << RGB << std::endl;

        newimage2 << type << std::endl;
        newimage2 << width << " " << height << std::endl;
        newimage2 << RGB << std::endl;

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {
                int r, g, b;
                int rC, gC, bC;
                r = (int)std::round(std::abs(matrix(i, j).r.real() / konst));
                g = (int)std::round(std::abs(matrix(i, j).g.real() / konst));
                b = (int)std::round(std::abs(matrix(i, j).b.real() / konst));
                if (r > 255) { r = 255; }
                if (g > 255) { g = 255; }
                if (b > 255) { b = 255; }
                if (r < 0) { r = 0; }
                if (g < 0) { g = 0; }
                if (b < 0) { b = 0; }
                
                newimage1 << r << " " << g << " " << b << std::endl;

                rC = (int)std::round(std::abs(matrix(i, j).r.imag()) / konst);
                gC = (int)std::round(std::abs(matrix(i, j).g.imag()) / konst);
                bC = (int)std::round(std::abs(matrix(i, j).b.imag()) / konst);

                if (rC > 255) { rC = 255; }
                if (gC > 255) { gC = 255; }
                if (bC > 255) { bC = 255; }
                if (rC < 0) { rC = 0; }
                if (gC < 0) { gC = 0; }
                if (bC < 0) { bC = 0; }

                newimage2 << rC << " " << gC << " " << bC << std::endl;

            }
        }
        newimage1.close();
        newimage2.close();
        

    }

    void MakePicturesFromMatrix2(std::string ImageNameNew1, std::string ImageNameNew2, Matrix<PixelC> matrix) {
        int konst = 10;
        std::ofstream newimage1;
        std::ofstream newimage2;
        newimage1.open(ImageNameNew1);
        newimage2.open(ImageNameNew2);

        newimage1 << type << std::endl;
        newimage1 << width << " " << height << std::endl;
        newimage1 << RGB << std::endl;

        newimage2 << type << std::endl;
        newimage2 << width << " " << height << std::endl;
        newimage2 << RGB << std::endl;

        int rMax = std::numeric_limits<int>::min(), gMax = std::numeric_limits<int>::min(), bMax = std::numeric_limits<int>::min();
        int rMin = std::numeric_limits<int>::max(), gMin = std::numeric_limits<int>::max(), bMin = std::numeric_limits<int>::max();
        int rcMax = std::numeric_limits<int>::min(), gcMax = std::numeric_limits<int>::min(), bcMax = std::numeric_limits<int>::min();
        int rcMin = std::numeric_limits<int>::max(), gcMin = std::numeric_limits<int>::max(), bcMin = std::numeric_limits<int>::max();

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {

                rMax = std::max(rMax, static_cast<int>(matrix(i, j).r.real()));
                gMax = std::max(gMax, static_cast<int>(matrix(i, j).g.real()));
                bMax = std::max(bMax, static_cast<int>(matrix(i, j).b.real()));

                rMin = std::min(rMin, static_cast<int>(matrix(i, j).r.real()));
                gMin = std::min(gMin, static_cast<int>(matrix(i, j).g.real()));
                bMin = std::min(bMin, static_cast<int>(matrix(i, j).b.real()));

                rcMax = std::max(rcMax, static_cast<int>(matrix(i, j).r.imag()));
                gcMax = std::max(gcMax, static_cast<int>(matrix(i, j).g.imag()));
                bcMax = std::max(bcMax, static_cast<int>(matrix(i, j).b.imag()));

                rcMin = std::min(rcMin, static_cast<int>(matrix(i, j).r.imag()));
                gcMin = std::min(gcMin, static_cast<int>(matrix(i, j).g.imag()));
                bcMin = std::min(bcMin, static_cast<int>(matrix(i, j).b.imag()));
            }
        }
        
        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {
                int r, g, b;
                int rc, gc, bc;
                r = static_cast<int>((matrix(i, j).r.real() - rMin) / (rMax - rMin) * 255);
                g = static_cast<int>((matrix(i, j).r.real() - rMin) / (rMax - rMin) * 255);
                b = static_cast<int>((matrix(i, j).r.real() - rMin) / (rMax - rMin) * 255);
                //if (r > 255) { r = 255; }
                //if (g > 255) { g = 255; }
                //if (b > 255) { b = 255; }

                newimage1 << r << " " << g << " " << b << std::endl;

                rc = static_cast<int>((matrix(i, j).r.imag() - rcMin) / (rcMax - rcMin) * 255);
                gc = static_cast<int>((matrix(i, j).g.imag() - gcMin) / (gcMax - gcMin) * 255);
                bc = static_cast<int>((matrix(i, j).b.imag() - bcMin) / (bcMax - bcMin) * 255);

                newimage2 << rc << " " << gc << " " << bc << std::endl;

            }
        }
        newimage1.close();
        newimage2.close();
    }
    */
    
    void MakePicFromMatrix(std::string ImageNameNew, Matrix<Pixel>& matrix) {
        std::ofstream newimage;
        newimage.open(ImageNameNew);

        newimage << type << std::endl;
        newimage << width << " " << height << std::endl;
        newimage << RGB << std::endl;

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {
                newimage << matrix(i, j).r << " " << matrix(i, j).g << " " << matrix(i, j).b << std::endl;
            }
        }
        newimage.close();

    }

    char map_to_ascii(int gray_value) {

        std::string characters = "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\",^`'";
        int num_chars = characters.length();
        int grayscale_range = 255;
        double step = (double)grayscale_range / num_chars;

        int index = gray_value / step;
        
        index = std::min(std::max(index, 0), num_chars - 1);

        return characters[index];
    }

    void ConvertToAsciiImage(std::string ImageName){
        std::ofstream newimage;
        newimage.open(ImageName);

        for (int i = 0; i < MatrixGray.getRows(); i++) {
            for (int j = 0; j < MatrixGray.getColumns(); j++) {
                newimage << map_to_ascii(MatrixGray(i, j)); 
            }
            newimage << std::endl;
        }
        newimage.close();
    }

    void DownSamplingAverage(int times,int koef) {
        for(int i = 0; i < times; ++i){
            MatrixGray.resizeToNearestTwo();
            Matrix<double> MatrixDownScaledGray;
            MatrixDownScaledGray.resize(MatrixGray.getRows() / koef, MatrixGray.getColumns() / koef);
            for (int m = 0; m < MatrixDownScaledGray.getRows(); m++) {
                for (int n = 0; n < MatrixDownScaledGray.getColumns(); n++) {
                    int sum = 0;
                    for (int p = 0; p < koef; p++) {
                        for (int q = 0; q < koef; q++) {
                            sum += MatrixGray(m * koef + p, n * koef + q);
                        }
                    }
                    MatrixDownScaledGray(m, n) = sum / (koef * koef);
                }
            }
            MatrixGray = MatrixDownScaledGray;
        }
    }

    void ToGrayScale(){
        MatrixGray.resize(std::stoi(height), std::stoi(width));
        for (int i = 0; i < MatrixGray.getRows(); i++) {
            for (int j = 0; j < MatrixGray.getColumns(); j++) {
                MatrixGray(i,j) = round(0.299 * MatrixPix(i,j).r + 0.587 * MatrixPix(i,j).g + 0.114 * MatrixPix(i,j).b);
            }
        }
    }

    void ColorFilter(int r, int g, int b){
        for (int i = 0; i < MatrixPix.getRows(); i++) {
            for (int j = 0; j < MatrixPix.getColumns(); j++) {
                MatrixPix(i,j).r += r;
                MatrixPix(i,j).g += g;
                MatrixPix(i,j).b += b;
                MatrixPix(i,j).r = MatrixPix(i,j).r > 255 ? 255 : MatrixPix(i,j).r;
                MatrixPix(i,j).g = MatrixPix(i,j).g > 255 ? 255 : MatrixPix(i,j).g;
                MatrixPix(i,j).b = MatrixPix(i,j).b > 255 ? 255 : MatrixPix(i,j).b;
            }
        }
    }

    void MakeGrayPic(std::string ImageNameNew, Matrix<double>& matrix) {
        std::ofstream newimage;
        newimage.open(ImageNameNew);
        newimage << "P2" << std::endl;
        newimage << matrix.getColumns() << " " << matrix.getRows() << std::endl;
        newimage << RGB << std::endl;

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {
                newimage << matrix(i, j) << std::endl;
            }
        }
        newimage.close();

    }

    /*
    void loadImg(std::string newImageName) {

        std::ifstream image;
        std::ofstream newimage;
        image.open(ImageName);
        newimage.open(newImageName);

        // copy
        std::string type, width, height, RGB;
        image >> type >> width >> height >> RGB;

        newimage << type << std::endl;
        newimage << width << " " << height << std::endl;
        newimage << RGB << std::endl;

        int r, g, b;
        while (image >> r >> g >> b) {

            if (r + 50 > 255) {
                r = 255;
            }
            else {
                r += 50;
            }
            if (g + 50 > 255) {
                g = 255;
            }
            else {
                g += 50;
            }
            if (b + 50 > 255) {
                b = 255;
            }
            else {
                b += 80;
            }

            newimage << r << " " << g << " " << b << std::endl;
        }
        image.close();
        newimage.close();
    }
    */
};
