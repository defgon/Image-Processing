# Image Processing with PPM

This project provides basic image processing functionalities for working with **PPM (Portable Pixmap)** image files.

## Features

The following functions are implemented in `Image.h`:

### Image Structure
- **Pixel (struct)**: Represents a pixel with RGB values.
- **PixelD (struct)**: Represents a pixel with double precision values.

### Image Processing Functions
- **sinc function**: Implements the sinc function as `sin(PI * x) / (PI * x)`.
- *(Additional functions will be listed here once extracted from the implementation.)*

## Dependencies

- Standard C++ Libraries (`iostream`, `fstream`, `vector`, etc.)
- `complex` (for complex number operations)
- Custom `matrix.h` (included in the project)

## Usage

To use the image processing functions, include `Image.h` in your project:

```cpp
#include "Image.h"

int main() {
    // Example usage here
}
```

## License

This project is open-source and available under the MIT License.
