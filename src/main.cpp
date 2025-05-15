#include "HatchDetectorApp.hpp"
#include <iostream>

int main()
{
    HatchDetectorApp detector;
    if (!detector.initialize())
        return 1;

    detector.run();
    return 0;
}