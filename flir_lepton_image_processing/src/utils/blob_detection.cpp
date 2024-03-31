#include "utils/blob_detection.h"

namespace flir_lepton {
namespace flir_lepton_image_processing {

  void BlobDetection::detectBlobs(const cv::Mat& inImage,
    std::vector<cv::KeyPoint>* keyPointsOut)
  {
    #ifdef DEBUG_TIME
    Timer::start("detectBlobs", "findHoles");
    #endif

    cv::SimpleBlobDetector::Params params;

    // Set various parameters
    params.minThreshold = Parameters::Blob::min_threshold;
    params.maxThreshold = Parameters::Blob::max_threshold;
    params.thresholdStep = Parameters::Blob::threshold_step;
    params.minArea = Parameters::Blob::min_area;
    params.maxArea = Parameters::Blob::max_area;
    params.minConvexity = Parameters::Blob::min_convexity;
    params.maxConvexity = Parameters::Blob::max_convexity;
    params.minInertiaRatio = Parameters::Blob::min_inertia_ratio;
    params.maxCircularity = Parameters::Blob::max_circularity;
    params.minCircularity = Parameters::Blob::min_circularity;
    params.filterByColor = Parameters::Blob::filter_by_color;
    params.filterByCircularity = Parameters::Blob::filter_by_circularity;

    // Corrected: Use cv::SimpleBlobDetector::create to obtain a pointer to a SimpleBlobDetector
    cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create(params);

    std::vector<cv::KeyPoint> keyPoints;

    // Use the pointer to call detect method
    blobDetector->detect(inImage, keyPoints);

    for (size_t keypointId = 0; keypointId < keyPoints.size(); keypointId++) {
      // if the keypoint is out of image limits, discard it
      if (keyPoints[keypointId].pt.x < inImage.cols &&
          keyPoints[keypointId].pt.x >= 0 &&
          keyPoints[keypointId].pt.y < inImage.rows &&
          keyPoints[keypointId].pt.y >= 0) {
        keyPointsOut->push_back(keyPoints[keypointId]);
      }
    }

    #ifdef DEBUG_TIME
    Timer::tick("detectBlobs");
    #endif
  }

}  // namespace flir_lepton_image_processing
}  // namespace flir_lepton
