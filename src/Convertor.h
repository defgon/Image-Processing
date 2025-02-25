#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <vector>


struct PixelChar {
    unsigned char r, g, b;
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
    inputFile.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(PixelChar));

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
