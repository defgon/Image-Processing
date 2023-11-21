#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include "matrix.h"
#include <vector>


struct Pixel {
    unsigned char r, g, b;
};

struct Image {

    std::string ImageName;

    Image (std::string Image) : ImageName(Image) {}

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
            // Apply the blue filter
            if (r + 50 > 255) {
                r = 255;
            }
            else {
                r += 50;
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
    std::vector<Pixel> pixels(width * height);
    inputFile.ignore();
    inputFile.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(Pixel));

    // Write to the ASCII PPM file
    outputFile << "P3\n" << width << " " << height << "\n255\n";
    for (const Pixel& pixel : pixels) {
        outputFile << static_cast<int>(pixel.r) << " "
            << static_cast<int>(pixel.g) << " "
            << static_cast<int>(pixel.b) << "\n";
    }

    inputFile.close();
    outputFile.close();
    std::cout << "Conversion successful!" << std::endl;

}

int main() {

    Image a("smpls\\moose.ppm");
    Image b("smpls\\papousek.ppm");
    Image c("smpls\\motyl.ppm");
    a.loadImg("img\\mooseB.ppm");
    b.loadImg("img\\papousekB.ppm");
    c.loadImg("img\\motylB.ppm");

    a.showImage();
    b.showImage();
    c.showImage();

    Image a1("img\\mooseB.ppm");
    Image b1("img\\papousekB.ppm");
    Image c1("img\\motylB.ppm");

    a1.showImage();
    b1.showImage();
    c1.showImage();

    return 0;
}


