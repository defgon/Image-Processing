#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cmath>
#include <iostream>

template <typename T>
class Matrix
{
private:
    int rows = 0;
    int cols = 0;
    std::vector<T> data;

public:
    Matrix() = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(const Matrix& rhs) = default;
    Matrix& operator=(Matrix&&) = default;
    Matrix(const Matrix& src) = default;

    Matrix(int rows, int cols)
        : rows(rows), cols(cols)
    {
        data.resize(rows * cols);
    }

    ~Matrix() {
        data.clear();
    }

    int getRows() const
    {
        return rows;
    }

    void resize(int rows_, int cols_)
    {
        data.clear();
        rows = rows_;
        cols = cols_;
        data.resize(rows * cols);
    }


    void print() const
    {
        for (int i = 0;
            i < this->getRows() * this->getColumns();
            i++)
            std::cout << this->getData()[i] << " ";
        std::cout << std::endl;
    }

    std::vector<T> getIthRow(int i) const
    {
        std::vector<T> ret;
        for(int w = 0; w < this->getColumns();++w)
            ret.push_back(this->getData()[i * cols + w]);
        return ret;
    }

    std::vector<T> getIthColumn(int i) const {
        std::vector<T> ret;
        for (int h = 0; h < this->getRows(); ++h)
            ret.push_back(this->getData()[h * cols + i]);
        return ret;
    }

    void setIthRow(int i, std::vector<T> dat)
    {
        for (int w = 0; w < this->getColumns(); ++w)
            data[i*cols + w] = dat[w];
    }

    std::vector<T> operator[](int index) {
        if (index < 0 || index >= rows) {
            throw std::out_of_range("Index out of bounds");
        }

        std::vector<T> rowData;
        for (int j = 0; j < cols; ++j) {
            rowData.push_back(data[index * cols + j]);
        }
        return rowData;
    }

    void zeros() {
        for (int i = 0;
            i < this->getRows() * this->getColumns();
            i++)
                this->getData()[i] = 0;
    }

    int getColumns() const
    {
        return cols;
    }

    std::vector<int> getDim() const
    {
        return std::vector<int>{this->getRows(),
            this->getColumns()};
    }

    T& operator()(int i, int j)
    {
        return data[i * cols + j];
    }

    const T& operator()(int i, int j) const
    {
        return data[i * cols + j];
    }

    const T* getData() const
    {
        return data.data();
    }

    int nearestPowerOf2(int n) const {
        return pow(2, int(ceil(log2(n))));
    }

    // Function to resize matrix to the nearest power of 2
    void resizeToNearestPowerOf2() {
        int newRows = nearestPowerOf2(rows);
        int newCols = nearestPowerOf2(cols);

        std::vector<T> newData; // Create an empty vector

        // Resize the vector to the required size
        newData.resize(static_cast<size_t>(newRows * newCols));

        for (int i = 0; i < std::min(rows, newRows); ++i) {
            for (int j = 0; j < std::min(cols, newCols); ++j) {
                newData[static_cast<size_t>(i * newCols + j)] = data[static_cast<size_t>(i * cols + j)];
            }
        }

        data = std::move(newData);
        rows = newRows;
        cols = newCols;
    }


    void rescaleToOriginalSize(int originalRows, int originalCols) {
        std::vector<T> newData; // Create an empty vector

        // Resize the vector to the required size
        newData.resize(static_cast<size_t>(originalRows * originalCols));

        for (int i = 0; i < originalRows; ++i) {
            for (int j = 0; j < originalCols; ++j) {
                if (i < rows && j < cols) {
                    newData[static_cast<size_t>(i * originalCols + j)] = data[static_cast<size_t>(i * cols + j)];
                }
            }
        }

        data = std::move(newData);
        rows = originalRows;
        cols = originalCols;
    }

};

// operator* číslo
template <typename T>
Matrix<T> operator*(T alfa, Matrix<T>& a)
{
    Matrix<T> c(a.getRows(), a.getColumns());
    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getColumns(); j++) {
            c(i, j) = alfa * a(i, j);
        }
    }
    return c;
}

template <typename T>
Matrix<T> operator*(Matrix<T>& a, T alfa)
{
    return alfa * a;
}

// operator+ číslo
template <typename T>
Matrix<T> operator+(T alfa, Matrix<T>& a)
{
    Matrix<T> c(a.getRows(), a.getColumns());
    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getColumns(); j++) {
            c(i, j) = alfa + a(i, j);
        }
    }
    return c;
}

template <typename T>
Matrix<T> operator+(Matrix<T>& a, T alfa)
{
    return alfa + a;
}

// operator- číslo
template <typename T>
Matrix<T> operator-(T alfa, Matrix<T>& a)
{
    Matrix<T> c(a.getRows(), a.getColumns());
    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getColumns(); j++) {
            c(i, j) = alfa - a(i, j);
        }
    }
    return c;
}

template <typename T>
Matrix<T> operator-(Matrix<T>& a, T alfa)
{
    return -(alfa)+a;
}

// operator/ číslo
template <typename T>
Matrix<T> operator/(T alfa, Matrix<T>& a)
{
    Matrix<T> c(a.getRows(), a.getColumns());
    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getColumns(); j++) {
            c(i, j) = alfa / a(i, j);
        }
    }
    return c;
}

template <typename T>
Matrix<T> operator/(Matrix<T>& a, T alfa)
{
    Matrix<T> c(a.getRows(), a.getColumns());
    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getColumns(); j++) {
            c(i, j) = a(i, j) / alfa;
        }
    }
    return c;
}

// operator+ Matice | Matice

template <typename T>
Matrix<T> operator+(Matrix<T>& a, Matrix<T>& b)
{
    if (a.getDim() == b.getDim())
    {
        Matrix<T> c(a.getRows(), a.getColumns());
        for (int i = 0; i < a.getColumns(); i++) {
            for (int j = 0; j <= a.getRows(); j++) {
                c(i, j) = a(i, j) + b(i, j);
            }
        }
        return c;
    }
    else {
        throw std::invalid_argument("Dimensions dont match!");
    }
}

// operator- Matice | Matice

template <typename T>
Matrix<T> operator-(Matrix<T>& a, Matrix<T>& b)
{
    if (a.getDim() == b.getDim())
    {
        Matrix<T> c(a.getRows(), a.getColumns());
        for (int i = 0; i < a.getColumns(); i++) {
            for (int j = 0; j <= a.getRows(); j++) {
                c(i, j) = a(i, j) - b(i, j);
            }
        }
        return c;
    }
    else {
        throw std::invalid_argument("Dimensions dont match!");
    }
}

// operator* Matice | Matice

template <typename T>
Matrix<T> operator*(Matrix<T>& a, Matrix<T>& b)
{
    if (a.getColumns() == b.getRows()) {
        Matrix<T> c(b.getRows(), a.getColumns());
        for (int i = 0; i < a.getColumns(); i++)
        {
            for (int j = 0; j < b.getRows(); j++)
            {
                for (int l = 0; l < a.getColumns(); l++)
                {
                    c(i, j) = c(i, j) + a(i, l) * b(l, j);
                }
            }
        }
        return c;
    }
    else {
        throw std::invalid_argument("Right matrix columns, "
            "dont equal left matrix rows!");
    }
}


template <int rows, int cols, typename T>
class StaticMatrix
{
private:
    T data[rows][cols];

public:
};


#endif // MATRIX_H
