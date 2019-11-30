#include <iostream>
#include "math_obj.hpp"

#define X_RES 1000
#define Y_RES 1000
#define RES_SIZE Vec2D(X_RES, Y_RES)
#define ENV_SIZE Vec2D(1, 1)

#define dt 1e-3    //timestep
#define mu 18.5e-6 //viscosity of air

Vec2D spacing = ENV_SIZE / RES_SIZE;

Vec2D getPosition(int i, int j)
{
    return spacing * Vec2D(i, j) + spacing / 2.0;
}

int main()
{
    double t = 0;
    VectorPlane<X_RES, Y_RES> velocity;
    VectorPlane<X_RES, Y_RES> externalAcceleration;
    ScalarPlane<X_RES, Y_RES> pressure;
    ScalarPlane<X_RES, Y_RES> massDensity;
    for (int i = 0; i < X_RES; ++i)
    {
        for (int j = 0; j < Y_RES; ++j)
        {
            velocity(i, j, Vec2D(0, 0));
            externalAcceleration(i, j, Vec2D(0, 0));
            massDensity(i, j, 1);
            pressure(i, j, getPosition(i, j).normSq());
        }
    }

    do
    {
        std::cout << t << std::endl;
        velocity << velocity + ((
                (VectorPlane<X_RES, Y_RES>(pressure.gradient(spacing)) * -1) +
                (velocity.laplacian(spacing) * mu) +
                (VectorPlane<X_RES, Y_RES>(velocity.divergence(spacing).gradient(spacing)) * (mu / 3.0)) +
                (externalAcceleration * massDensity)
            ) * (massDensity^-1) +
            (
                VectorPlane<X_RES, Y_RES>(
                    velocity * VectorPlane<X_RES, Y_RES>(velocity.getX().gradient(spacing)),
                    velocity * VectorPlane<X_RES, Y_RES>(velocity.getY().gradient(spacing))
                ) * -1
            )
        ) * dt;
        t += dt;
    } while (t <= 1.0);
}