#include "Eigen.h"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

typedef pair<int, Point2d> Observation;

typedef vector<Observation> Observations;

struct MapPoint {
  Vector3d point;
  Observations observations;
};

typedef std::vector<MapPoint> Map;
