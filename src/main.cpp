#include <iostream>
#include "Convertor.h"
#include "Image.h"


int main() {
    
    // initialize variables
    std::string image_name,image_name_after,ascii_image_name,color_image_name;
    int option,option_fourier,option_color,down_sample_konst,radius;
    int r,g,b;

    //start program
    std::cout << "Program started succesfully! \n";
    std::cout << "Enter your image name: (folder/file_name.ppm) \n";
    std::cout << "Enter image name: ";
    std::cin >> image_name;
    std::cout << "\n";
    Image a(image_name);
    
    //options of image processing
    std::cout << "Choose from options: \n";
    std::cout << " 1 = fourier transform filtering \n";
    std::cout << " 2 = convert your image to ascii \n";
    std::cout << " 3 = color filters \n";
    std::cout << " 4 = gray downsampling \n";
    std::cout << "Enter option: ";
    std::cin >> option;
    std::cout << "\n";

    switch (option)
    {
    //fourier transform filtering
    case 1:
        a.fft2D(a.CMatrixPix, true);
        a.fftshift2D(a.CMatrixPix);
        std::cout << "What to filter ?\n";
        std::cout << "Choose from options: \n";
        std::cout << " 1 = high frequencies \n";
        std::cout << " 2 = low frequencies \n";
        std::cout << "Enter option: ";
        std::cin >> option_fourier;
        std::cout << "\n";
        switch (option_fourier)
        {
        case 1:
            std::cout << "Enter radius " << "(smaller than" << a.CMatrixPix.getColumns() << "x" << a.CMatrixPix.getRows() << "):";
            std::cin >> radius;
            std::cout << "\n";
            a.Frequencyfilter(radius,false);
            break;
        case 2:
            std::cout << "Enter radius " << "(smaller than" << a.CMatrixPix.getColumns() << "x" << a.CMatrixPix.getRows() << "):";
            std::cin >> radius;
            std::cout << "\n";
            a.Frequencyfilter(radius,true);
            break;
        
        default:
            std::cout << "Invalid input (input must be number from 1 to 4)\n";
            break;
        }
        a.fftshift2D(a.CMatrixPix);
        a.fft2D(a.CMatrixPix, false);
        std::cout << "Enter filtered image name (folder/file_name.ppm): ";
        std::cin >> image_name_after;
        std::cout << "\n";
        a.ImageIFFT(image_name_after,1);
        break;
    //converting image to ascii .txt format
    case 2:
        std::cout << "Enter downsizing scale (for example 2 for 2x smaller): ";
        std::cin >> down_sample_konst;
        std::cout << "\n";
        std::cout << "Enter image name (folder/file_name.txt):";
        std::cin >> ascii_image_name;
        std::cout << "\n";
        a.DownSamplingAverage(down_sample_konst,2);
        a.ConvertToAsciiImage(ascii_image_name);
        break;
    //color filtering image
    case 3:
        std::cout << "What to filter ?\n";
        std::cout << "Enter red value u want to add (up to 255 for effect): ";
        std::cin >> r;
        std::cout << "\n";
        std::cout << "Enter green value u want to add (up to 255 for effect): ";
        std::cin >> g;
        std::cout << "\n";
        std::cout << "Enter blue value u want to add (up to 255 for effect):";
        std::cin >> b;
        std::cout << "\n";
        std::cout << "Enter image name (folder/file_name.ppm):";
        std::cin >> color_image_name;
        std::cout << "\n";
        a.ColorFilter(r,g,b);
        a.MakePicFromMatrix(color_image_name,a.MatrixPix);
        break;
    //downsizing grayscale image
    case 4:
        std::cout << "Enter downsizing scale (for example 2 for 2x smaller): ";
        std::cin >> down_sample_konst;
        std::cout << "\n";
        std::cout << "Enter name of image after downsizing (folder/file_name.pgm): ";
        std::cin >> image_name_after;
        std::cout << "\n";
        a.DownSamplingAverage(down_sample_konst,2);
        a.MakeGrayPic(image_name_after,a.MatrixGray);
        break;
    
    default:
        std::cout << "Invalid input (input must be number from 1 to 4)\n";
        break;
    }

    return 0;
    
}
