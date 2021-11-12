//
// Created by Collin Sinclair on 11/6/21.
//

#include "tests.h"
#include <iostream>
using namespace std;
int main()
{
    bool forceMagTest = test_forceMagnitude();
    if (forceMagTest)
    {
        cout << "Force magnitude test passed" << endl;
    }
    else
    {
        cout << "Force magnitude test failed" << endl;
    }
    bool magTest = test_magnitude();
    if (magTest)
    {
        cout << "Magnitude test passed" << endl;
    }
    else
    {
        cout << "Magnitude test failed" << endl;
    }
    bool unitDirVecTest = test_unitDirectionVector();
    if (unitDirVecTest)
    {
        cout << "Unit direction vector test passed" << endl;
    }
    else
    {
        cout << "Unit direction vector test failed" << endl;
    }
    bool forceVecTest = test_forceVector();
    if (forceVecTest)
    {
        cout << "Force vector test passed" << endl;
    }
    else
    {
        cout << "Force vector test failed" << endl;
    }
    bool calcForceVecTest = test_calculateForceVectors();
    if (calcForceVecTest)
    {
        cout << "Calc force vector test passed" << endl;
    }
    else
    {
        cout << "Calc force vector test failed" << endl;
    }
    
    return 0;
}