#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "Eigen.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
//#include "FreeImageHelper.h"

typedef unsigned char BYTE;

using namespace cv;

// reads sensor files according to
// https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor {
public:
  VirtualSensor(int increment = 10)
      : m_currentIdx(-1), m_increment(increment) {}

  //  ~VirtualSensor() {
  //    SAFE_DELETE_ARRAY(m_depthFrame);
  //    SAFE_DELETE_ARRAY(m_grayscaleFrame);
  //  }

  bool Init(const std::string &datasetDir) {
    m_baseDir = datasetDir;

    // read filename lists
    if (!ReadFileList(datasetDir + "depth.txt", m_filenameDepthImages,
                      m_depthImagesTimeStamps))
      return false;
    if (!ReadFileList(datasetDir + "rgb.txt", m_filenameColorImages,
                      m_colorImagesTimeStamps))
      return false;

    if (m_filenameDepthImages.size() != m_filenameColorImages.size())
      return false;

    // image resolutions
    m_imageWidth = 640;
    m_imageHeight = 480;
    m_depthImageWidth = 640;
    m_depthImageHeight = 480;

    // intrinsics
    m_intrinsics << 525.0f, 0.0f, 319.5f, 0.0f, 525.0f, 239.5f, 0.0f, 0.0f,
        1.0f;

    m_depthIntrinsics = m_intrinsics;

    m_extrinsics.setIdentity();
    m_depthExtrinsics.setIdentity();

    //    m_depthFrame = new float[m_depthImageWidth * m_depthImageHeight];
    //    for (unsigned int i = 0; i < m_depthImageWidth * m_depthImageHeight;
    //    ++i)
    //      m_depthFrame[i] = 0.5f;

    //    m_colorFrame = new BYTE[4 * m_colorImageWidth * m_colorImageHeight];
    //    for (unsigned int i = 0; i < 4 * m_colorImageWidth *
    //    m_colorImageHeight;
    //         ++i)
    //      m_colorFrame[i] = 255;

    m_currentIdx = -1;
    return true;
  }

  bool ProcessNextFrame() {
    if (m_currentIdx == -1)
      m_currentIdx = 0;
    else
      m_currentIdx += m_increment;

    if ((unsigned int)m_currentIdx >=
        (unsigned int)m_filenameColorImages.size())
      return false;

    std::cout << "ProcessNextFrame [" << m_currentIdx << " | "
              << m_filenameColorImages.size() << "]" << std::endl;

    m_grayscaleFrame = imread(
        samples::findFile(m_baseDir + m_filenameColorImages[m_currentIdx]),
        IMREAD_GRAYSCALE);
    //    memcpy(m_colorFrame, rgbImage.data, 4 * 640 * 480);

    m_depthFrame = imread(
        samples::findFile(m_baseDir + m_filenameDepthImages[m_currentIdx]),
        IMREAD_UNCHANGED);

    // depth images are scaled by 5000 (see
    // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
    for (auto &val : m_depthFrame) {
      if (val == 0)
        val = MINF;
      else
        val = val * 1.0f / 5000.0f;
    }

    return true;
  }

  std::string GetCurrentRGBTimestamp() {
    return m_filenameColorImages[m_currentIdx].substr(
        4, m_filenameColorImages[m_currentIdx].size() - 8);
  }

  std::string GetCurrentDepthTimestamp() {
    return m_filenameDepthImages[m_currentIdx].substr(
        6, m_filenameDepthImages[m_currentIdx].size() - 10);
  }

  unsigned int GetCurrentFrameCnt() { return (unsigned int)m_currentIdx; }

  // get current color data
  Mat GetGrayscaleFrame() { return m_grayscaleFrame; }
  // get current depth data
  Mat GetDepthFrame() { return m_depthFrame; }

  // color camera info
  Eigen::Matrix3f GetCameraIntrinsics() { return m_intrinsics; }

  Eigen::Matrix4f GetCameraExtrinsics() { return m_extrinsics; }

  unsigned int GetImageWidth() { return m_imageWidth; }

  unsigned int GetImageHeight() { return m_imageHeight; }

  // depth (ir) camera info
  Eigen::Matrix3f GetDepthIntrinsics() { return m_depthIntrinsics; }

  Eigen::Matrix4f GetDepthExtrinsics() { return m_depthExtrinsics; }

  unsigned int GetDepthImageWidth() { return m_imageWidth; }

  unsigned int GetDepthImageHeight() { return m_imageHeight; }

private:
  bool ReadFileList(const std::string &filename,
                    std::vector<std::string> &result,
                    std::vector<double> &timestamps) {
    std::ifstream fileDepthList(filename, std::ios::in);
    if (!fileDepthList.is_open())
      return false;
    result.clear();
    timestamps.clear();
    std::string dump;
    std::getline(fileDepthList, dump);
    std::getline(fileDepthList, dump);
    std::getline(fileDepthList, dump);
    while (fileDepthList.good()) {
      double timestamp;
      fileDepthList >> timestamp;
      std::string filename;
      fileDepthList >> filename;
      if (filename == "")
        break;
      timestamps.push_back(timestamp);
      result.push_back(filename);
    }
    fileDepthList.close();
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // current frame index
  int m_currentIdx;

  int m_increment;

  // frame data
  Mat_<float> m_depthFrame;
  Mat m_grayscaleFrame;

  // color camera info
  Eigen::Matrix3f m_intrinsics;
  Eigen::Matrix4f m_extrinsics;
  unsigned int m_imageWidth;
  unsigned int m_imageHeight;

  // depth (ir) camera info
  Eigen::Matrix3f m_depthIntrinsics;
  Eigen::Matrix4f m_depthExtrinsics;
  unsigned int m_depthImageWidth;
  unsigned int m_depthImageHeight;

  // base dir
  std::string m_baseDir;
  // filenamelist depth
  std::vector<std::string> m_filenameDepthImages;
  std::vector<double> m_depthImagesTimeStamps;
  // filenamelist color
  std::vector<std::string> m_filenameColorImages;
  std::vector<double> m_colorImagesTimeStamps;
};
