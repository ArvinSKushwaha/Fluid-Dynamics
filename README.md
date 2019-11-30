# Fluid-Dynamics
This is code for a 2-D (extensible to 3-D) fluid simulator I am working on as a parallel project to the [Plasma Simulator][https://github.com/ArvinSKushwaha/Plasma-Simulator]

This repository will also contain a RCNN (Recurrent Convolutional Neural Network) that can be trained to run simulations of fluids.

* math_obj.hpp: Contains useful mathematical objects
    * Vec3D: A vector class in 3D space
    * Vec2D: A vector class in 2D space
    * SizeTuple: An unsigned integer class (to hold matrix sizes)
    * ScalarField: A 3D matrix (field) of scalars; it is a discretized representation of a scalar field function
    * VectorField: The Vector analog of the ScalarField. Represents a discretized representation of a vector field function.
    * ScalarPlane: A 2D matrix (field) of scalars; Is the 2D analog of the ScalarField
    * VectorPlane: A 2D matrix (field) of vectors; Is the Vector analog of the ScalarPlane
* main.cpp: The main C++ file containing the fluid simulation code
* main.py: Will contain the RCNN upon completion