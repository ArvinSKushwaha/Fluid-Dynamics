#include <iostream>
#include "math_obj_f.hpp"

#define X_RES 10
#define Y_RES 10
#define Z_RES 10
#define RES_SIZE Vec3D(X_RES, Y_RES, Z_RES)
#define ENV_SIZE Vec3D(1, 1, 1)

#define dt 1e-3    //timestep
#define mu 18.5e-6 //viscosity of air

Vec3D spacing = ENV_SIZE / RES_SIZE;

Vec3D getPosition(int i, int j, int k)
{
    return spacing * Vec3D(i, j, k) + spacing / 2.0;
}

int main()
{
    std::cout << "VectorField<" << X_RES << ", " << Y_RES << ", " << Z_RES << ">: " << sizeof(VectorField<X_RES, Y_RES, Z_RES>) << " bytes" << std::endl;
    std::cout << "ScalarField<" << X_RES << ", " << Y_RES << ", " << Z_RES << ">: " << sizeof(ScalarField<X_RES, Y_RES, Z_RES>) << " bytes" << std::endl;
    std::cout << "Vec3D: " << sizeof(Vec3D) << " bytes" << std::endl;
    std::cout << "Vec2D: " << sizeof(Vec2D) << " bytes" << std::endl;
    std::cout << "double: " << sizeof(double) << " bytes" << std::endl;
    std::cout << "float: " << sizeof(float) << " bytes" << std::endl;
    std::cout << "uint: " << sizeof(uint) << " bytes" << std::endl;
    std::cout << "int: " << sizeof(int) << " bytes" << std::endl;
    double t = 0;
    VectorField<X_RES, Y_RES, Z_RES> velocity;
    VectorField<X_RES, Y_RES, Z_RES> externalAcceleration;
    ScalarField<X_RES, Y_RES, Z_RES> pressure;
    ScalarField<X_RES, Y_RES, Z_RES> massDensity;

    #pragma omp parallel for collapse(3) schedule(static, 4000)
    for (int i = 0; i < X_RES; ++i)
    {
        for (int j = 0; j < Y_RES; ++j)
        {
            for(int k = 0; k < Z_RES; ++k)
            {
                velocity(i, j, k, Vec3D(0, 0, 0));
                externalAcceleration(i, j, k, Vec3D(0, 0, 0));
                massDensity(i, j, k, 1);
                pressure(i, j, k, getPosition(i, j, k).normSq());
            }
        }
    }
    
    #pragma omp parallel for schedule(static, 4000)
    for(int iterations = 0; iterations < 1000; iterations++)
    {
        std::cout << iterations << std::endl;
        velocity += ((
                (VectorField<X_RES, Y_RES, Z_RES>(pressure.gradient(spacing)) * -1) +
                (velocity.laplacian(spacing) * mu) +
                (VectorField<X_RES, Y_RES, Z_RES>(velocity.divergence(spacing).gradient(spacing)) * (mu / 3.0)) +
                (externalAcceleration * massDensity)
            ) * (massDensity^-1) +
            (
                VectorField<X_RES, Y_RES, Z_RES>(
                    velocity * VectorField<X_RES, Y_RES, Z_RES>(velocity.getX().gradient(spacing)),
                    velocity * VectorField<X_RES, Y_RES, Z_RES>(velocity.getY().gradient(spacing)),
                    velocity * VectorField<X_RES, Y_RES, Z_RES>(velocity.getZ().gradient(spacing))
                ) * -1
            )
        ) * dt;
        t += dt;
        velocity.toFile("output/"+std::to_string(iterations) + "output");
    }
}