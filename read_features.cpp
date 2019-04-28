#include"cnpy.h"
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>

int main()
{
    //load the entire npz file
    cnpy::npz_t my_npz = cnpy::npz_load("features.npz");
    cnpy::NpyArray arr_mv1 = my_npz["features"];
    double* mv1 = arr_mv1.data<double>();
    for(int i = 0; i < 20; i++) {
        std::cout << mv1[i] << std::endl;
    }
}
