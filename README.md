# Image Processing in C++

This project is an image processing tool implemented in C++ that allows users to apply various transformations to images, including Fourier Transform filtering, ASCII conversion, color filtering, and grayscale downsampling. It processes images using a matrix-based approach and supports both binary and ASCII PPM formats.

## Features

- **Fourier Transform Filtering**: Apply high-pass and low-pass frequency filters to images.
- **ASCII Conversion**: Convert images into ASCII representations.
- **Color Filtering**: Adjust RGB color channels.
- **Grayscale Downsampling**: Reduce image size while preserving grayscale values.

## Installation

To compile the project, use a C++ compiler that supports C++11 or later. The recommended way to build the project is using CMake:

```sh
mkdir build
cd build
cmake ..
make
```

## Usage

Run the executable and follow the interactive prompts:

```sh
./image_processor
```

### Available Operations

1. **Fourier Transform Filtering**
   - Apply high-frequency or low-frequency filters.
   - Saves the filtered image.

2. **ASCII Image Conversion**
   - Converts the image into an ASCII text representation.

3. **Color Filtering**
   - Adjusts RGB values of an image and saves the output.

4. **Grayscale Downsampling**
   - Reduces the resolution of the grayscale image.

## File Structure

- `main.cpp` - Main entry point of the application, handling user input.
- `Image.h` - Core image processing functions.
- `matrix.h` - Matrix operations used in image transformations.
- `Convertor.h` - Handles binary-to-ASCII PPM conversion.

## Example Commands

After running `./image_processor`, users will be prompted to enter the image file path and choose an operation. Example workflow:

1. Enter the image name: `images/sample.ppm`
2. Choose an option (e.g., `1` for Fourier Transform Filtering)
3. Enter the required parameters (e.g., filter radius for Fourier filtering)
4. Provide an output file name: `output/sample_filtered.ppm`

## Dependencies

This project does not require any external dependencies beyond the C++ standard library.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
