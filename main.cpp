#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include "matrix.h"
#include <vector>

const double PI = 3.14159265358979323846;

struct PixelChar {
    unsigned char r, g, b;
};

struct Pixel {
    int r, g, b;
};


double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

struct Image {
    Matrix<Pixel> MatrixPix;
    Matrix<Pixel> ReMatrixPix;
    Matrix<Pixel> ImMatrixPix;
    std::string type, width, height, RGB;
    std::string ImageName;
    Image (std::string Image) : ImageName(Image) {
        ImgToMatrix();
    }

    int showImage() {
        // Read the PPM image
        cv::Mat image = cv::imread(ImageName, cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cout << "Could not open or find the image!\n";
            return -1;
        }

        // Display the image
        cv::imshow("Generated Image", image);
        cv::waitKey(0);
        return 0;
    }

    void makeRandomImage(int w, int h, int RGBmax) {
        std::ofstream image;
        image.open(ImageName);

        srand(time(0));

        if (image.is_open()) {
            image << "P3" << std::endl;
            image << w << " " << h << std::endl;
            image << RGBmax << std::endl;

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    image << 255 << " " << 0 << " " << 0 << std::endl;
                }
            }
        }
    }

    void ImgToMatrix() {
        std::ifstream image;
        image.open(ImageName);

        image >> type >> width >> height >> RGB;
        MatrixPix.resize(std::stoi(width), std::stoi(height));
        
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

    void DFT() {
        ReMatrixPix.resize(std::stoi(width), std::stoi(height));
        ImMatrixPix.resize(std::stoi(width), std::stoi(height));

        for (int u = 0; u < MatrixPix.getRows(); ++u) {
            for (int v = 0; v < MatrixPix.getColumns(); ++v) {
                for (int i = 0; i < MatrixPix.getRows(); ++i) {
                    for (int j = 0; j < MatrixPix.getColumns(); ++j) {
                        double angle = 2 * PI * ((double)(u * i) / MatrixPix.getRows() + (double)(v * j) / MatrixPix.getColumns());
                        std::complex<double> Xr = (double)MatrixPix(i, j).r * std::complex<double>(std::cos(angle), -std::sin(angle));
                        std::complex<double> Xg = (double)MatrixPix(i, j).g * std::complex<double>(std::cos(angle), -std::sin(angle));
                        std::complex<double> Xb = (double)MatrixPix(i, j).b * std::complex<double>(std::cos(angle), -std::sin(angle));
                        ReMatrixPix(u, v).r += Xr.real();
                        ReMatrixPix(u, v).g += Xg.real();
                        ReMatrixPix(u, v).b += Xb.real();

                        ImMatrixPix(u, v).r += Xr.imag();
                        ImMatrixPix(u, v).g += Xg.imag();
                        ImMatrixPix(u, v).b += Xb.imag();

                    }
                }
            }
            std::cout << u << std::endl;
        }

        for (int u = 0; u < MatrixPix.getRows(); ++u) {
            for (int v = 0; v < MatrixPix.getColumns(); ++v) {
                ReMatrixPix(u, v).r = (int)std::round(255*sigmoid(ReMatrixPix(u, v).r));
                ReMatrixPix(u, v).g = (int)std::round(255*sigmoid(ReMatrixPix(u, v).g));
                ReMatrixPix(u, v).b = (int)std::round(255*sigmoid(ReMatrixPix(u, v).b));
                if (ReMatrixPix(u, v).r > 255) { ReMatrixPix(u, v).r = 255; }
                if (ReMatrixPix(u, v).g > 255) { ReMatrixPix(u, v).g = 255; }
                if (ReMatrixPix(u, v).b > 255) { ReMatrixPix(u, v).b = 255; }

                ImMatrixPix(u, v).r = (int)std::round(255 * sigmoid(ImMatrixPix(u, v).r));
                ImMatrixPix(u, v).g = (int)std::round(255 * sigmoid(ImMatrixPix(u, v).g));
                ImMatrixPix(u, v).b = (int)std::round(255 * sigmoid(ImMatrixPix(u, v).b));
                if (ImMatrixPix(u, v).r > 255) { ImMatrixPix(u, v).r = 255; }
                if (ImMatrixPix(u, v).g > 255) { ImMatrixPix(u, v).g = 255; }
                if (ImMatrixPix(u, v).b > 255) { ImMatrixPix(u, v).b = 255; }
            }
        }
    }

    void IDFT(std::string imageReStr, std::string imageImStr) {
        Matrix<Pixel> ReMatrixPix;
        Matrix<Pixel> ImMatrixPix;
        Matrix<Pixel> MatrixPix;
        std::ifstream imageRe;
        std::ifstream imageIm;
        imageRe.open(imageReStr);
        imageIm.open(imageImStr);

        std::string type, width, height, RGB;
        imageRe >> type >> width >> height >> RGB;
        imageIm >> type >> width >> height >> RGB;

        ReMatrixPix.resize(std::stoi(width), std::stoi(height));
        ImMatrixPix.resize(std::stoi(width), std::stoi(height));
        MatrixPix.resize(std::stoi(width), std::stoi(height));

        int r, g, b;
        for (int u = 0; u < ReMatrixPix.getRows(); ++u) {
            for (int v = 0; v < ReMatrixPix.getColumns(); ++v) {
                imageRe >> r >> g >> b;
                ReMatrixPix(u, v).r = r;
                ReMatrixPix(u, v).g = g;
                ReMatrixPix(u, v).b = b;

                imageIm >> r >> g >> b;
                ImMatrixPix(u, v).r = r;
                ImMatrixPix(u, v).g = g;
                ImMatrixPix(u, v).b = b;
            }
        }

        for (int i = 0; i < ReMatrixPix.getRows(); ++i) {
            for (int j = 0; j < ReMatrixPix.getColumns(); ++j) {
                for (int u = 0; u < ReMatrixPix.getRows(); ++u) {
                    for (int v = 0; v < ReMatrixPix.getColumns(); ++v) {

                        double angle = 2 * PI * ((double)(u * i) / ReMatrixPix.getRows() + (double)(v * j) / ReMatrixPix.getColumns());
                        std::complex<double> complexExp(std::cos(angle), std::sin(angle));

                        // Accumulate contributions from frequency domain
                        MatrixPix(i, j).r += ReMatrixPix(u, v).r * complexExp.real() + ImMatrixPix(u, v).r * complexExp.imag();
                        MatrixPix(i, j).g += ReMatrixPix(u, v).g * complexExp.real() + ImMatrixPix(u, v).g * complexExp.imag();
                        MatrixPix(i, j).b += ReMatrixPix(u, v).b * complexExp.real() + ImMatrixPix(u, v).b * complexExp.imag();

                        MatrixPix(i, j).r = MatrixPix(i, j).r / (ReMatrixPix.getRows() * ReMatrixPix.getColumns());
                        MatrixPix(i, j).g = MatrixPix(i, j).g / (ReMatrixPix.getRows() * ReMatrixPix.getColumns());
                        MatrixPix(i, j).b = MatrixPix(i, j).b / (ReMatrixPix.getRows() * ReMatrixPix.getColumns());
                    }
                }
            }
            std::cout << i << std::endl;
        }

        for (int i = 0; i < MatrixPix.getRows(); ++i) {
            for (int j = 0; j < MatrixPix.getColumns(); ++j) {
                ReMatrixPix(i, j).r = (int)std::round(255 * sigmoid(ReMatrixPix(i, j).r));
                ReMatrixPix(i, j).g = (int)std::round(255 * sigmoid(ReMatrixPix(i, j).g));
                ReMatrixPix(i, j).b = (int)std::round(255 * sigmoid(ReMatrixPix(i, j).b));
                if (ReMatrixPix(i, j).r > 255) { ReMatrixPix(i, j).r = 255; }
                if (ReMatrixPix(i, j).g > 255) { ReMatrixPix(i, j).g = 255; }
                if (ReMatrixPix(i, j).b > 255) { ReMatrixPix(i, j).b = 255; }
            }
        }

    }

    void MakePicFromMatrix(std::string ImageNameNew,Matrix<Pixel>& matrix) {
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
};


void convertBinaryToASCII(const std::string& inputFilename, const std::string& outputFilename) {
    
    std::ifstream inputFile(inputFilename, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file!" << std::endl;
        std::cout << "Conversion failed!" << std::endl;
    }

    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        std::cout << "Conversion failed!" << std::endl;
    }

    std::string type;
    int width, height, maxColor;
    inputFile >> type >> width >> height >> maxColor;

    // Check if the file is in the correct PPM format
    if (type != "P6" || maxColor > 255) {
        std::cerr << "Unsupported PPM format or color depth!" << std::endl;
        std::cout << "Conversion failed!" << std::endl;
    }

    // Read pixel data from the binary file
    std::vector<PixelChar> pixels(width * height);
    inputFile.ignore();
    inputFile.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(Pixel));

    // Write to the ASCII PPM file
    outputFile << "P3\n" << width << " " << height << "\n255\n";
    for (const PixelChar& pixel : pixels) {
        outputFile << static_cast<int>(pixel.r) << " "
            << static_cast<int>(pixel.g) << " "
            << static_cast<int>(pixel.b) << "\n";
    }

    inputFile.close();
    outputFile.close();
    std::cout << "Conversion successful!" << std::endl;

}


int main() {
    /*Image a("smpls\\parot.ppm");
    a.DFT();

    a.MakePicFromMatrix("dft\\parotRe.ppm",a.ReMatrixPix);
    a.MakePicFromMatrix("dft\\parotIm.ppm", a.ImMatrixPix);*/

    /*Image i("dft\\parotRe.ppm");
    i.IDFT("dft\\parotRe.ppm", "dft\\parotIm.ppm");
    i.MakePicFromMatrix("smpls\\IDFT.ppm", i.MatrixPix);*/

    Image a("smpls\\IDFT.ppm");
    a.showImage();

    return 0;
}
