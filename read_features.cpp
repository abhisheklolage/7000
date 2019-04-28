#include"cnpy.h"
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>

#define endl std::endl
#define cout std::cout

int main(int argc, char* argv[]) {
    if(argc == 1) {
        cout << "Enter feature file path (npz)" << endl;
        return 1;
    }
    //load the entire npz file
    cout << "Loading features from file " << argv[1] << endl;
    cnpy::npz_t my_npz = cnpy::npz_load(argv[1]);
    cnpy::NpyArray arr_mv1 = my_npz["features"];
    double* mv1 = arr_mv1.data<double>();
    for(int i = 0; i < 20; i++) {
        cout << mv1[i] << endl;
    }
}
