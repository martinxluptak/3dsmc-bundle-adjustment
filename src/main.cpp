#include <iostream>

#include "../headers/VirtualSensor.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
  std::string filenameIn =
      std::string("../../Data/rgbd_dataset_freiburg1_xyz/");
  std::string filenameBaseOut = std::string("mesh_");

  // Load video
  std::cout << "Initialize virtual sensor..." << std::endl;
  VirtualSensor sensor;
  if (!sensor.Init(filenameIn)) {
    std::cout << "Failed to initialize the sensor!\nCheck file path!"
              << std::endl;
    return -1;
  }

  // We store a first frame as a reference frame. All next frames are tracked
  // relatively to the first frame.
  sensor.ProcessNextFrame();

  //  for (unsigned int i = 0;
  //       i < 4 * sensor.GetColorImageWidth() * sensor.GetColorImageHeight();
  //       ++i)
  //    std::cout << (int)sensor.GetColorRGBX()[i] << std::endl;
}
