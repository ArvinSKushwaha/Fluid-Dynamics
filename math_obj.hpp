#include <iostream>
#include <cmath>
#include <fstream>
#include <any>
#include <cstring>
#include <array>
#include <vector>

#define INF_VEC Vec3D(INFINITY, INFINITY, INFINITY)
#define ZERO_VEC Vec3D(0, 0, 0)
#define NAN_VEC Vec3D(NAN, NAN, NAN);
#define X_HAT Vec3D(1, 0, 0)
#define Y_HAT Vec3D(0, 1, 0)
#define Z_HAT Vec3D(0, 0, 1)
#define PI M_PI                        // dimensionless
#define LIGHT_SPEED 299792458.0        // m s^-1
#define AVOGADRO 6.02214086e+23        // dimensionless
#define E_CHARGE 1.602176634 * 1e-19   // s A
#define M_PROTON 1.67262192369 * 1e-27 // kg
#define M_ELECTRON 9.10938356e-31      // kg
#define MU_0 PI * 4e-7                 // m kg s^-2 A^-2
#define EPSILON_0 8.85418782e-12       // m^-3 kg^-1 s^4 A^2
#define k_B 1.38064852e-23             // m^2 kg s^-2 K^-1

// A vector function class
class Vec3D
{
public:
    double x, y, z;
    Vec3D(double x_, double y_, double z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }

    Vec3D()
    {
        x = 0;
        y = 0;
        z = 0;
    }

    Vec3D operator-=(Vec3D n)
    {
        this->x - n.x;
        this->y - n.y;
        this->z - n.z;
        return *this;
    }

    Vec3D operator+=(Vec3D n)
    {
        this->x + n.x;
        this->y + n.y;
        this->z + n.z;
        return *this;
    }

    Vec3D operator-(Vec3D n)
    {
        return add(n * -1);
    }

    Vec3D operator-()
    {
        return *this * -1;
    }

    Vec3D operator+(Vec3D n)
    {
        return add(n);
    }

    Vec3D operator*(Vec3D n)
    {
        return multiply(n);
    }

    Vec3D operator*=(Vec3D n)
    {
        this->x *n.x;
        this->y *n.y;
        this->z *n.z;
        return *this;
    }

    Vec3D operator*(double n)
    {
        return multiply(n);
    }

    Vec3D operator*=(double n)
    {
        this->x *n;
        this->y *n;
        this->z *n;
        return *this;
    }

    Vec3D operator/(double n)
    {
        return multiply(1.0 / n);
    }

    Vec3D operator/(Vec3D n)
    {
        return multiply(Vec3D(1.0/n.x, 1.0/n.y, 1.0/n.z));
    }

    double operator%(Vec3D n)
    {
        return dot(n);
    }

    Vec3D operator()(double x, double y, double z)
    {
        Vec3D d = Vec3D(x, y, z);
        return d;
    }

    bool operator==(Vec3D n)
    {
        return n.x == x && n.y == y && n.z == z;
    }

    double dot(Vec3D n)
    {
        return x * n.x + y * n.y + z * n.z;
    }

    Vec3D cross(Vec3D n)
    {
        return Vec3D(y * n.z - z * n.y, z * n.x - x * n.z, x * n.y - y * n.x);
    }

    Vec3D add(Vec3D n)
    {
        return Vec3D(x + n.x, y + n.y, z + n.z);
    }

    Vec3D add(double n)
    {
        return Vec3D(x + n, y + n, z + n);
    }

    Vec3D multiply(Vec3D n)
    {
        return Vec3D(x * n.x, y * n.y, z * n.z);
    }

    Vec3D multiply(double n)
    {
        return Vec3D(x * n, y * n, z * n);
    }

    std::string cstr()
    {
        return "x: " + std::to_string(x) + "\r\ny: " + std::to_string(y) + "\r\nz: " + std::to_string(z);
    }

    std::string vstr()
    {
        return "<" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ">";
    }

    std::string sstr()
    {
        return std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z);
    }

    std::string cstr(Vec3D n)
    {
        return "x: " + std::to_string(n.x) + "\r\ny: " + std::to_string(n.y) + "\r\nz: " + std::to_string(n.z);
    }

    std::string vstr(Vec3D n)
    {
        return "<" + std::to_string(n.x) + ", " + std::to_string(n.y) + ", " + std::to_string(n.z) + ">";
    }

    std::string sstr(Vec3D n)
    {
        return std::to_string(n.x) + " " + std::to_string(n.y) + " " + std::to_string(n.z);
    }

    Vec3D sinVec(Vec3D v)
    {
        return Vec3D(sin(v.x), sin(v.y), sin(v.z));
    }

    Vec3D cosVec(Vec3D v)
    {
        return Vec3D(cos(v.x), cos(v.y), cos(v.z));
    }

    Vec3D tanVec(Vec3D v)
    {
        return Vec3D(tan(v.x), tan(v.y), tan(v.z));
    }

    double norm(Vec3D n)
    {
        return sqrt(n.dot(n));
    }

    double norm()
    {
        return sqrt(x * x + y * y + z * z);
    }

    double normSq()
    {
        return x * x + y * y + z * z;
    }

    Vec3D normalize(Vec3D n)
    {
        return n / n.norm();
    }

    Vec3D normalize()
    {
        double l = norm();
        x /= l;
        y /= l;
        z /= l;
        return *this;
    }

    Vec3D transform(double matrix[3][3])
    {
        double x_ = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z;
        double y_ = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z;
        double z_ = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z;
        return Vec3D(x_, y_, z_);
    }

    Vec3D operator*(double matrix[3][3])
    {
        return transform(matrix);
    }

    Vec3D operator*=(double matrix[3][3])
    {
        return this->transform(matrix);
    }
};

class Vec2D
{
public:
    double x, y;
    Vec2D(double x_, double y_)
    {
        x = x_;
        y = y_;
    }

    Vec2D()
    {
        x = 0;
        y = 0;
    }

    Vec2D operator-=(Vec2D n)
    {
        this->x - n.x;
        this->y - n.y;
        return *this;
    }

    Vec2D operator+=(Vec2D n)
    {
        this->x + n.x;
        this->y + n.y;
        return *this;
    }

    Vec2D operator-(Vec2D n)
    {
        return add(n * -1);
    }

    Vec2D operator-()
    {
        return *this * -1;
    }

    Vec2D operator+(Vec2D n)
    {
        return add(n);
    }

    Vec2D operator*(Vec2D n)
    {
        return multiply(n);
    }

    Vec2D operator*=(Vec2D n)
    {
        this->x *n.x;
        this->y *n.y;
        return *this;
    }

    Vec2D operator*(double n)
    {
        return multiply(n);
    }

    Vec2D operator*=(double n)
    {
        this->x *n;
        this->y *n;
        return *this;
    }

    Vec2D operator/(double n)
    {
        return multiply(1.0 / n);
    }

    Vec2D operator/(Vec2D n)
    {
        return multiply(Vec2D(1.0/n.x, 1.0/n.y));
    }

    double operator%(Vec2D n)
    {
        return dot(n);
    }

    Vec2D operator()(double x, double y)
    {
        Vec2D d = Vec2D(x, y);
        return d;
    }

    bool operator==(Vec2D n)
    {
        return n.x == x && n.y == y;
    }

    double dot(Vec2D n)
    {
        return x * n.x + y * n.y;
    }

    Vec2D add(Vec2D n)
    {
        return Vec2D(x + n.x, y + n.y);
    }

    Vec2D add(double n)
    {
        return Vec2D(x + n, y + n);
    }

    Vec2D multiply(Vec2D n)
    {
        return Vec2D(x * n.x, y * n.y);
    }

    Vec2D multiply(double n)
    {
        return Vec2D(x * n, y * n);
    }

    std::string cstr()
    {
        return "x: " + std::to_string(x) + "\r\ny: " + std::to_string(y);
    }

    std::string vstr()
    {
        return "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
    }

    std::string sstr()
    {
        return std::to_string(x) + " " + std::to_string(y);
    }

    std::string cstr(Vec2D n)
    {
        return "x: " + std::to_string(n.x) + "\r\ny: " + std::to_string(n.y);
    }

    std::string vstr(Vec2D n)
    {
        return "<" + std::to_string(n.x) + ", " + std::to_string(n.y) + ">";
    }

    std::string sstr(Vec2D n)
    {
        return std::to_string(n.x) + " " + std::to_string(n.y);
    }

    Vec2D sinVec(Vec2D v)
    {
        return Vec2D(sin(v.x), sin(v.y));
    }

    Vec2D cosVec(Vec2D v)
    {
        return Vec2D(cos(v.x), cos(v.y));
    }

    Vec2D tanVec(Vec2D v)
    {
        return Vec2D(tan(v.x), tan(v.y));
    }

    double norm(Vec2D n)
    {
        return sqrt(n.dot(n));
    }

    double norm()
    {
        return sqrt(x * x + y * y);
    }

    double normSq()
    {
        return x * x + y * y;
    }

    Vec2D normalize(Vec2D n)
    {
        return n / n.norm();
    }

    Vec2D normalize()
    {
        double l = norm();
        x /= l;
        y /= l;
        return *this;
    }

    Vec2D transform(double matrix[2][2])
    {
        double x_ = matrix[0][0] * x + matrix[0][1] * y;
        double y_ = matrix[1][0] * x + matrix[1][1] * y;
        return Vec2D(x_, y_);
    }

    Vec2D operator*(double matrix[2][2])
    {
        return transform(matrix);
    }

    Vec2D operator*=(double matrix[2][2])
    {
        return this->transform(matrix);
    }
};

// Holds the size (requires integers)
struct SizeTuple3D
{
    const int x, y, z;
};

struct SizeTuple2D
{
    const int x, y;
};

// Represents a 3-D field of Scalars
template <uint a, uint b, uint c>
class ScalarField
{
public:
    unsigned const int dims = 3;
    const unsigned int rows = a;
    const unsigned int cols = b;
    const unsigned int depth = c;
    std::array<unsigned int, 3> matrixSize = {rows, cols, depth};
    double *data = (double *)malloc(sizeof(double) * a * b * c);

    ScalarField(){};

    ScalarField<a, b, c> (double *field)
    {
        data = field;
    }

    int index(uint i, uint j, uint k)
    {
        return k + j * b + i * a * b;
    }

    double operator()(unsigned int i, unsigned int j, unsigned int k)
    {
        return data[index(i, j, k)];
    }

    void operator()(uint i, uint j, uint k, double v)
    {
        data[index(i, j, k)] = v;
    }

    ScalarField<a, b, c> operator+(double n)
    {
        ScalarField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) + n);
                }
            }
        }
        return N;
    }

    ScalarField<a, b, c> operator*(double n)
    {
        ScalarField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) * n);
                }
            }
        }
        return N;
    }

    ScalarField<a, b, c> operator+(ScalarField<a, b, c> A)
    {
        ScalarField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) + A(i, j, k));
                }
            }
        }
        return N;
    }

    ScalarField<a, b, c> operator*(ScalarField<a, b, c> A)
    {
        ScalarField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) * A(i, j, k));
                }
            }
        }
        return N;
    }

    ScalarField<a, b, c> operator+=(double n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] += n;
                }
            }
        }
        return *this;
    }

    ScalarField<a, b, c> operator*=(double n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] *= n;
                }
            }
        }
        return *this;
    }

    ScalarField<a, b, c> operator+=(ScalarField<a, b, c> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] += A(i, j, k);
                }
            }
        }
        return *this;
    }

    ScalarField<a, b, c> operator*=(ScalarField<a, b, c> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] *= A(i, j, k);
                }
            }
        }
        return *this;
    }

    ScalarField<a, b, c> operator^(double n)
    {
        ScalarField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    if (operator()(i, j, k) == 0)
                    {
                        N(i, j, k, 0);
                    }
                    else
                    {
                        N(i, j, k, pow(operator()(i, j, k), n));
                    }
                }
            }
        }
        return N;
    }

    ScalarField<a, b, c> operator<<(double dataIn[a][b][c])
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, dataIn[i][j][k]);
                }
            }
        }
        return *this;
    }

    ScalarField<a, b, c> operator<<(std::array<std::array<std::array<double, c>, b>, a> dataIn)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, dataIn[i][j][k]);
                }
            }
        }
        return *this;
    }

    ScalarField<a, b, c> operator<<(ScalarField<a, b, c> dataIn)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, dataIn(i, j, k));
                }
            }
        }
        return *this;
    }

    std::array<std::array<std::array<Vec3D, c>, b>, a> gradient(Vec3D spacing)
    {
        std::array<std::array<std::array<Vec3D, c>, b>, a> Gradiented;
        double dFdx, dFdy, dFdz;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    if (i > 0 && j > 0 && k > 0 && i + 1 < matrixSize[0] && j + 1 < matrixSize[1] && k + 1 < matrixSize[2])
                    {
                        dFdx = (operator()(i + 1, j, k) - operator()(i - 1, j, k)) / (2 * spacing.x);
                        dFdy = (operator()(i, j + 1, k) - operator()(i, j - 1, k)) / (2 * spacing.y);
                        dFdz = (operator()(i, j, k + 1) - operator()(i, j, k - 1)) / (2 * spacing.z);
                    }
                    if (i == 0)
                    {
                        dFdx = (operator()(i + 1, j, k) - operator()(i, j, k)) / (spacing.x);
                    }
                    if (i + 1 == matrixSize[0])
                    {
                        dFdx = (operator()(i, j, k) - operator()(i - 1, j, k)) / (spacing.x);
                    }
                    if (k == 0)
                    {
                        dFdz = (operator()(i, j, k + 1) - operator()(i, j, k)) / (spacing.z);
                    }
                    if (k + 1 == matrixSize[2])
                    {
                        dFdz = (operator()(i, j, k) - operator()(i, j, k - 1)) / (spacing.z);
                    }
                    if (j == 0)
                    {
                        dFdy = (operator()(i, j + 1, k) - operator()(i, j, k)) / (spacing.y);
                    }
                    if (j + 1 == matrixSize[1])
                    {
                        dFdy = (operator()(i, j, k) - operator()(i, j - 1, k)) / (spacing.y);
                    }
                    Gradiented[i][j][k] = Vec3D(dFdx, dFdy, dFdz);
                }
            }
        }
        return Gradiented;
    }

    void toFile(std::string filename)
    {
        std::ofstream myfile(filename);
        if (myfile.is_open())
        {
            myfile << matrixSize[0] << "\n";
            myfile << matrixSize[1] << "\n";
            myfile << matrixSize[2] << "\n";

            for (int i = 0; i < matrixSize[0]; ++i)
            {
                for (int j = 0; j < matrixSize[1]; ++j)
                {
                    for (int k = 0; k < matrixSize[2]; ++k)
                    {
                        myfile << i << " " << j << " " << k << " " << operator()(i, j, k) << "\n";
                    }
                }
            }
            myfile.close();
        }
        else
        {
            std::cout << filename + " was not openable" << std::endl;
        }
    }
};

// Represents a 3-D field of Vec3D
template <uint a, uint b, uint c>
class VectorField
{
public:
    unsigned const int dims = 3;
    const unsigned int rows = a;
    const unsigned int cols = b;
    const unsigned int depth = c;
    std::array<unsigned int, 3> matrixSize = {rows, cols, depth};
    Vec3D *data = (Vec3D *)malloc(sizeof(Vec3D) * a * b * c);

    VectorField(){};

    VectorField<a, b, c> (Vec3D *field)
    {
        data = field;
    }

    VectorField<a, b, c> (ScalarField<a, b, c> A, ScalarField<a, b, c> B, ScalarField<a, b, c> C)
    {
        for(int i = 0; i < matrixSize[0]; ++i)
        {
            for(int j = 0; j < matrixSize[1]; ++j)
            {
                for(int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, Vec3D(A(i, j, k), B(i, j, k), C(i, j, k)));
                }
            }
        }
    }

    int index(uint i, uint j, uint k)
    {
        return k + j * b + i * a * b;
    }

    Vec3D operator()(unsigned int i, unsigned int j, unsigned int k)
    {
        return data[index(i, j, k)];
    }

    void operator()(uint i, uint j, uint k, Vec3D v)
    {
        data[index(i, j, k)] = v;
    }

    ScalarField<a, b, c> getX()
    {
        ScalarField<a, b, c> X;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    X(i, j, k, operator()(i, j, k).x);
                }
            }
        }
        return X;
    }

    ScalarField<a, b, c> getY()
    {
        ScalarField<a, b, c> Y;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    Y(i, j, k, operator()(i, j, k).y);
                }
            }
        }
        return Y;
    }

    ScalarField<a, b, c> getZ()
    {
        ScalarField<a, b, c> Z;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    Z(i, j, k, operator()(i, j, k).z);
                }
            }
        }
        return Z;
    }

    VectorField<a, b, c> operator+(Vec3D n)
    {
        VectorField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) + n);
                }
            }
        }
        return N;
    }

    VectorField<a, b, c> operator*(double n)
    {
        VectorField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) * n);
                }
            }
        }
        return N;
    }

    VectorField<a, b, c> operator+(VectorField<a, b, c> A)
    {
        VectorField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) + A(i, j, k));
                }
            }
        }
        return N;
    }

    VectorField<a, b, c> operator*(ScalarField<a, b, c> A)
    {
        VectorField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k) * A(i, j, k));
                }
            }
        }
        return N;
    }

    ScalarField<a, b, c> operator*(VectorField<a, b, c> A)
    {
        ScalarField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k).dot(A(i, j, k)));
                }
            }
        }
        return N;
    }

    VectorField<a, b, c> operator+=(Vec3D n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] += n;
                }
            }
        }
        return *this;
    }

    VectorField<a, b, c> operator*=(double n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] *= n;
                }
            }
        }
        return *this;
    }

    VectorField<a, b, c> operator+=(VectorField<a, b, c> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] += A(i, j, k);
                }
            }
        }
        return *this;
    }

    VectorField<a, b, c> operator*=(ScalarField<a, b, c> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    data[index(i, j, k)] *= A(i, j, k);
                }
            }
        }
        return *this;
    }

    ScalarField<a, b, c> operator*(Vec3D A)
    {
        ScalarField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k).dot(A(i, j, k)));
                }
            }
        }
        return N;
    }

    VectorField<a, b, c> operator<<(Vec3D dataIn[a][b][c])
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, dataIn[i][j][k]);
                }
            }
        }
        return *this;
    }

    VectorField<a, b, c> operator<<(std::array<std::array<std::array<Vec3D, c>, b>, a> dataIn)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, dataIn[i][j][k]);
                }
            }
        }
        return *this;
    }

    VectorField<a, b, c> operator<<(ScalarField<a, b, c> fields[3])
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, Vec3D(fields[0](i, j, k), fields[1](i, j, k), fields[2](i, j, k)));
                }
            }
        }
        return *this;
    }

    VectorField<a, b, c> operator<<(VectorField<a, b, c> field)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    operator()(i, j, k, field(i, j, k));
                }
            }
        }
        return *this;
    }

    VectorField<a, b, c> cross(VectorField<a, b, c> A)
    {
        VectorField<a, b, c> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, k, operator()(i, j, k).cross(A(i, j, k)));
                }
            }
        }
        return N;
    }

    VectorField<a, b, c> curl(Vec3D spacing)
    {
        double dXdy, dXdz, dYdx, dYdz, dZdx, dZdy;
        VectorField<a, b, c> Curled;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    if (i > 0 && j > 0 && k > 0 && i + 1 < matrixSize[0] && j + 1 < matrixSize[1] && k + 1 < matrixSize[2])
                    {
                        dXdy = (operator()(i, j + 1, k).x - operator()(i, j - 1, k).x) / (2 * spacing.y);
                        dXdz = (operator()(i, j, k + 1).x - operator()(i, j, k - 1).x) / (2 * spacing.z);
                        dYdx = (operator()(i + 1, j, k).y - operator()(i - 1, j, k).y) / (2 * spacing.x);
                        dYdz = (operator()(i, j, k + 1).y - operator()(i, j, k - 1).y) / (2 * spacing.z);
                        dZdx = (operator()(i + 1, j, k).z - operator()(i - 1, j, k).z) / (2 * spacing.x);
                        dZdy = (operator()(i, j + 1, k).z - operator()(i, j - 1, k).z) / (2 * spacing.y);
                    }
                    if (i == 0)
                    {
                        dZdx = (operator()(i + 1, j, k).z - operator()(i, j, k).z) / (spacing.x);
                        dYdx = (operator()(i + 1, j, k).y - operator()(i, j, k).y) / (spacing.x);
                    }
                    if (i + 1 == matrixSize[0])
                    {
                        dZdx = (operator()(i, j, k).z - operator()(i - 1, j, k).z) / (spacing.x);
                        dYdx = (operator()(i, j, k).y - operator()(i - 1, j, k).y) / (spacing.x);
                    }
                    if (k == 0)
                    {
                        dXdz = (operator()(i, j, k + 1).x - operator()(i, j, k).x) / (spacing.z);
                        dYdz = (operator()(i, j, k + 1).y - operator()(i, j, k).y) / (spacing.z);
                    }
                    if (k + 1 == matrixSize[2])
                    {
                        dXdz = (operator()(i, j, k).x - operator()(i, j, k - 1).x) / (spacing.z);
                        dYdz = (operator()(i, j, k).y - operator()(i, j, k - 1).y) / (spacing.z);
                    }
                    if (j == 0)
                    {
                        dXdy = (operator()(i, j + 1, k).x - operator()(i, j, k).x) / (spacing.y);
                        dZdy = (operator()(i, j + 1, k).z - operator()(i, j, k).z) / (spacing.y);
                    }
                    if (j + 1 == matrixSize[1])
                    {
                        dXdy = (operator()(i, j, k).x - operator()(i, j - 1, k).x) / (spacing.y);
                        dZdy = (operator()(i, j, k).z - operator()(i, j - 1, k).z) / (spacing.y);
                    }
                    Curled(i, j, k, Vec3D(dZdy - dYdz, dXdz - dZdx, dYdx - dXdy));
                }
            }
        }
        return Curled;
    }

    ScalarField<a, b, c> divergence(Vec3D spacing)
    {
        double dXdx, dYdy, dZdz;
        ScalarField<a, b, c> Diverged;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    if (i > 0 && j > 0 && k > 0 && i + 1 < matrixSize[0] && j + 1 < matrixSize[1] && k + 1 < matrixSize[2])
                    {
                        dXdx = (operator()(i + 1, j, k).x - operator()(i - 1, j, k).x) / (2 * spacing.x);
                        dYdy = (operator()(i, j + 1, k).y - operator()(i, j - 1, k).y) / (2 * spacing.y);
                        dZdz = (operator()(i, j, k + 1).z - operator()(i, j, k - 1).z) / (2 * spacing.z);
                    }
                    if (i == 0)
                    {
                        dXdx = (operator()(i + 1, j, k).x - operator()(i, j, k).x) / (spacing.x);
                    }
                    if (i + 1 == matrixSize[0])
                    {
                        dXdx = (operator()(i, j, k).x - operator()(i - 1, j, k).x) / (spacing.x);
                    }
                    if (k == 0)
                    {
                        dZdz = (operator()(i, j, k + 1).z - operator()(i, j, k).z) / (spacing.z);
                    }
                    if (k + 1 == matrixSize[2])
                    {
                        dZdz = (operator()(i, j, k).z - operator()(i, j, k - 1).z) / (spacing.z);
                    }
                    if (j == 0)
                    {
                        dYdy = (operator()(i, j + 1, k).y - operator()(i, j, k).y) / (spacing.y);
                    }
                    if (j + 1 == matrixSize[1])
                    {
                        dYdy = (operator()(i, j, k).y - operator()(i, j - 1, k).y) / (spacing.y);
                    }
                    Diverged(i, j, k, dXdx + dYdy + dZdz);
                }
            }
        }
        return Diverged;
    }

    VectorField<a, b, c> laplacian(Vec3D spacing)
    {
        VectorField<a, b, c> N;
        ScalarField<a, b, c> A[3] = {getX().gradient(spacing).divergence(spacing), getY().gradient(spacing).divergence(spacing), getZ().gradient(spacing).divergence(spacing)};
        N << A;
        return N;
    }

    void toFile(std::string filename)
    {
        std::ofstream myfile(filename);
        if (myfile.is_open())
        {
            myfile << matrixSize[0] << "\n";
            myfile << matrixSize[1] << "\n";
            myfile << matrixSize[2] << "\n";

            for (int i = 0; i < matrixSize[0]; ++i)
            {
                for (int j = 0; j < matrixSize[1]; ++j)
                {
                    for (int k = 0; k < matrixSize[2]; ++k)
                    {
                        std::cout << i << " " << j << " " << k << " " << operator()(i, j, k).sstr() << "\n";
                        myfile << i << " " << j << " " << k << " " << operator()(i, j, k).sstr() << "\n";
                    }
                }
            }
            myfile.close();
        }
        else
        {
            std::cout << filename + " was not openable" << std::endl;
        }
    }
};

template <uint a, uint b>
class ScalarPlane
{
public:
    unsigned const int dims = 2;
    const unsigned int rows = a;
    const unsigned int cols = b;
    std::array<unsigned int, 2> matrixSize = {rows, cols};
    double *data = (double *)malloc(sizeof(double) * a * b);

    ScalarPlane(){};

    ScalarPlane<a, b> (double *field)
    {
        data = field;
    }

    int index(uint i, uint j)
    {
        return j + i * a;
    }

    double operator()(unsigned int i, unsigned int j)
    {
        return data[index(i, j)];
    }

    void operator()(uint i, uint j, double v)
    {
        data[index(i, j)] = v;
    }

    ScalarPlane<a, b> operator+(double n)
    {
        ScalarPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j) + n);
            }
        }
        return N;
    }

    ScalarPlane<a, b> operator*(double n)
    {
        ScalarPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j) * n);
            }
        }
        return N;
    }

    ScalarPlane<a, b> operator+(ScalarPlane<a, b> A)
    {
        ScalarPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j) + A(i, j));
            }
        }
        return N;
    }

    ScalarPlane<a, b> operator*(ScalarPlane<a, b> A)
    {
        ScalarPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j) * A(i, j));
            }
        }
        return N;
    }

    ScalarPlane<a, b> operator+=(double n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] += n;
            }
        }
        return *this;
    }

    ScalarPlane<a, b> operator*=(double n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] *= n;
            }
        }
        return *this;
    }

    ScalarPlane<a, b> operator+=(ScalarPlane<a, b> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] += A(i, j);
            }
        }
        return *this;
    }

    ScalarPlane<a, b> operator*=(ScalarPlane<a, b> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] *= A(i, j);
            }
        }
        return *this;
    }

    ScalarPlane<a, b> operator^(double n)
    {
        ScalarPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                if (operator()(i, j) == 0)
                {
                    N(i, j, 0);
                }
                else
                {
                    N(i, j, pow(operator()(i, j), n));
                }
            }
        }
        return N;
    }

    ScalarPlane<a, b> operator<<(double dataIn[a][b])
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, dataIn[i][j]);
            }
        }
        return *this;
    }

    ScalarPlane<a, b> operator<<(std::array<std::array<double, b>, a> dataIn)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, dataIn[i][j]);
            }
        }
        return *this;
    }

    ScalarPlane<a, b> operator<<(ScalarPlane<a, b> dataIn)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, dataIn(i, j));
            }
        }
        return *this;
    }

    Vec2D *gradient(Vec2D spacing)
    {
        Vec2D *Gradiented = (Vec2D*) malloc(sizeof(Vec2D) * matrixSize[0] * matrixSize[1]);
        double dFdx, dFdy;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                if (i > 0 && j > 0 && i + 1 < matrixSize[0] && j + 1 < matrixSize[1])
                {
                    dFdx = (operator()(i + 1, j) - operator()(i - 1, j)) / (2 * spacing.x);
                    dFdy = (operator()(i, j + 1) - operator()(i, j - 1)) / (2 * spacing.y);
                }
                if (i == 0)
                {
                    dFdx = (operator()(i + 1, j) - operator()(i, j)) / (spacing.x);
                }
                if (i + 1 == matrixSize[0])
                {
                    dFdx = (operator()(i, j) - operator()(i - 1, j)) / (spacing.x);
                }
                if (j == 0)
                {
                    dFdy = (operator()(i, j + 1) - operator()(i, j)) / (spacing.y);
                }
                if (j + 1 == matrixSize[1])
                {
                    dFdy = (operator()(i, j) - operator()(i, j - 1)) / (spacing.y);
                }
                Gradiented[index(i, j)] = Vec2D(dFdx, dFdy);
            }
        }
        return Gradiented;
    }

    void toFile(std::string filename)
    {
        std::ofstream myfile(filename);
        if (myfile.is_open())
        {
            myfile << matrixSize[0] << "\n";
            myfile << matrixSize[1] << "\n";

            for (int i = 0; i < matrixSize[0]; ++i)
            {
                for (int j = 0; j < matrixSize[1]; ++j)
                {
                    myfile << i << " " << j << " " << operator()(i, j) << "\n";
                }
            }
            myfile.close();
        }
        else
        {
            std::cout << filename + " was not openable" << std::endl;
        }
    }
};

template <uint a, uint b>
class VectorPlane
{
public:
    unsigned const int dims = 2;
    const unsigned int rows = a;
    const unsigned int cols = b;
    std::array<unsigned int, 3> matrixSize = {rows, cols};
    Vec2D *data = (Vec2D *)malloc(sizeof(Vec2D) * a * b);

    VectorPlane(){};

    VectorPlane<a, b> (Vec2D *field)
    {
        data = field;
    }

    VectorPlane<a, b> (ScalarPlane<a, b> A, ScalarPlane<a, b> B)
    {
        for(int i = 0; i < matrixSize[0]; ++i)
        {
            for(int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, Vec2D(A(i, j), B(i, j)));
            }
        }
    }

    int index(uint i, uint j)
    {
        return j + i * a;
    }

    Vec2D operator()(unsigned int i, unsigned int j)
    {
        return data[index(i, j)];
    }

    void operator()(uint i, uint j, Vec2D v)
    {
        data[index(i, j)] = v;
    }

    ScalarPlane<a, b> getX()
    {
        ScalarPlane<a, b> X;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                X(i, j, operator()(i, j).x);
            }
        }
        return X;
    }

    ScalarPlane<a, b> getY()
    {
        ScalarPlane<a, b> Y;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                Y(i, j, operator()(i, j).y);
            }
        }
        return Y;
    }

    VectorPlane<a, b> operator+(Vec2D n)
    {
        VectorPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j) + n);
            }
        }
        return N;
    }

    VectorPlane<a, b> operator*(double n)
    {
        VectorPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j) * n);
            }
        }
        return N;
    }

    VectorPlane<a, b> operator+(VectorPlane<a, b> A)
    {
        VectorPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j) + A(i, j));
            }
        }
        return N;
    }

    VectorPlane<a, b> operator*(ScalarPlane<a, b> A)
    {
        VectorPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                for (int k = 0; k < matrixSize[2]; ++k)
                {
                    N(i, j, operator()(i, j) * A(i, j));
                }
            }
        }
        return N;
    }

    ScalarPlane<a, b> operator*(VectorPlane<a, b> A)
    {
        ScalarPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, operator()(i, j).dot(A(i, j)));
            }
        }
        return N;
    }

    VectorPlane<a, b> operator+=(Vec2D n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] = data[index(i, j)] + n;
            }
        }
        return *this;
    }

    VectorPlane<a, b> operator*=(double n)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] = data[index(i, j)] * n;
            }
        }
        return *this;
    }

    VectorPlane<a, b> operator+=(VectorPlane<a, b> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] = operator()(i, j) + A(i, j);
            }
        }
        return *this;
    }

    VectorPlane<a, b> operator*=(ScalarPlane<a, b> A)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                data[index(i, j)] *= A(i, j);
            }
        }
        return *this;
    }

    ScalarPlane<a, b> operator*(Vec2D A)
    {
        ScalarPlane<a, b> N;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; j++)
            {
                N(i, j, operator()(i, j).dot(A(i, j)));
            }
        }
        return N;
    }

    VectorPlane<a, b> operator<<(Vec2D dataIn[a][b])
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, dataIn[i][j]);
            }
        }
        return *this;
    }

    VectorPlane<a, b> operator<<(std::array<std::array<Vec2D, b>, a> dataIn)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, dataIn[i][j]);
            }
        }
        return *this;
    }

    VectorPlane<a, b> operator<<(ScalarPlane<a, b> fields[2])
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, Vec2D(fields[0](i, j), fields[1](i, j)));
            }
        }
        return *this;
    }

    VectorPlane<a, b> operator<<(VectorPlane<a, b> field)
    {
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                operator()(i, j, field(i, j));
            }
        }
        return *this;
    }

    VectorPlane<a, b> operator()(std::array<std::array<Vec2D, b>, a> field)
    {
        VectorPlane<a, b> N;
        for(int i = 0; i < matrixSize[0]; ++i)
        {
            for(int j = 0; j < matrixSize[1]; ++j)
            {
                N(i, j, field[i][j]);
            }
        }
        return N;
    }

    ScalarPlane<a, b> divergence(Vec2D spacing)
    {
        double dXdx, dYdy;
        ScalarPlane<a, b> Diverged;
        for (int i = 0; i < matrixSize[0]; ++i)
        {
            for (int j = 0; j < matrixSize[1]; ++j)
            {
                if (i > 0 && j > 0 && i + 1 < matrixSize[0] && j + 1 < matrixSize[1])
                {
                    dXdx = (operator()(i + 1, j).x - operator()(i - 1, j).x) / (2 * spacing.x);
                    dYdy = (operator()(i, j + 1).y - operator()(i, j - 1).y) / (2 * spacing.y);
                }
                if (i == 0)
                {
                    dXdx = (operator()(i + 1, j).x - operator()(i, j).x) / (spacing.x);
                }
                if (i + 1 == matrixSize[0])
                {
                    dXdx = (operator()(i, j).x - operator()(i - 1, j).x) / (spacing.x);
                }
                if (j == 0)
                {
                    dYdy = (operator()(i, j + 1).y - operator()(i, j).y) / (spacing.y);
                }
                if (j + 1 == matrixSize[1])
                {
                    dYdy = (operator()(i, j).y - operator()(i, j - 1).y) / (spacing.y);
                }
                Diverged(i, j, dXdx + dYdy);
            }
        }
        return Diverged;
    }

    VectorPlane<a, b> laplacian(Vec2D spacing)
    {
        VectorPlane<a, b> N;
        ScalarPlane<a, b> A[2] = {VectorPlane<a, b>(getX().gradient(spacing)).divergence(spacing), VectorPlane<a, b>(getY().gradient(spacing)).divergence(spacing)};
        N << A;
        return N;
    }

    void toFile(std::string filename)
    {
        std::ofstream myfile(filename);
        if (myfile.is_open())
        {
            myfile << matrixSize[0] << "\n";
            myfile << matrixSize[1] << "\n";

            for (int i = 0; i < matrixSize[0]; ++i)
            {
                for (int j = 0; j < matrixSize[1]; ++j)
                {
                    myfile << i << " " << j << " " << operator()(i, j).sstr() << "\n";
                }
            }
            myfile.close();
        }
        else
        {
            std::cout << filename + " was not openable" << std::endl;
        }
    }
};