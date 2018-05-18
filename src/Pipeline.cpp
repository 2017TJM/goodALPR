//
// Created by 庾金科 on 23/10/2017.
//

#include "../include/Pipeline.h"
#include <boost/filesystem.hpp>
#include <string>
namespace fs = boost::filesystem;


namespace pr {



	const int HorizontalPadding = 4;
	PipelinePR::PipelinePR(std::string detector_filename,
		std::string finemapping_prototxt, std::string finemapping_caffemodel,
		std::string segmentation_prototxt, std::string segmentation_caffemodel,
		std::string charRecognization_proto, std::string charRecognization_caffemodel,
		std::string segmentationfree_proto, std::string segmentationfree_caffemodel) {
		plateDetection = new PlateDetection(detector_filename);
		fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
		plateSegmentation = new PlateSegmentation(segmentation_prototxt, segmentation_caffemodel);
		generalRecognizer = new CNNRecognizer(charRecognization_proto, charRecognization_caffemodel);
		segmentationFreeRecognizer = new SegmentationFreeRecognizer(segmentationfree_proto, segmentationfree_caffemodel);

	}

	PipelinePR::~PipelinePR() {

		delete plateDetection;
		delete fineMapping;
		delete plateSegmentation;
		delete generalRecognizer;
		delete segmentationFreeRecognizer;
	}

	std::vector<PlateInfo> PipelinePR::RunPiplineAsImage(cv::Mat plateImage, int method, std::string fileName) {
		std::vector<PlateInfo> results;
		std::vector<PlateInfo> plates;
		plateDetection->plateDetectionRough(plateImage, plates, fileName, 36, 700);
		std::string filePath2 = "C:\\Users\\Michael\\Desktop\\filePath2";
		std::string filePath3 = "C:\\Users\\Michael\\Desktop\\filePath3";
		std::string filePath4 = "C:\\Users\\Michael\\Desktop\\filePath4";
		std::string filePath5 = "C:\\Users\\Michael\\Desktop\\filePath5";
		std::string filePath1 = "C:\\Users\\Michael\\Desktop\\filePath1";
		if (!fs::exists(filePath1)) {
			fs::create_directories(filePath1);
		}
		if (!fs::exists(filePath2)) {
			fs::create_directories(filePath2);
		}
		if (!fs::exists(filePath3)) {
			fs::create_directories(filePath3);
		}
		if (!fs::exists(filePath4)) {
			fs::create_directories(filePath4);
		}
		if (!fs::exists(filePath5)) {
			fs::create_directories(filePath5);
		}
		for (PlateInfo plateinfo : plates) {
			cv::Mat image_finemapping = plateinfo.getPlateImage();
			std::string fileName1 = plateinfo.getFileName();
			cv::imwrite(filePath1 + "\\" + fileName1 + ".jpg", image_finemapping);
			image_finemapping = fineMapping->FineMappingVertical(image_finemapping);
			std::string fileName2 = fileName1 + "-" + std::to_string(0) + ".jpg";
			cv::imwrite(filePath2 + "\\" + fileName2, image_finemapping);
			image_finemapping = pr::fastdeskew(image_finemapping, 5);
			std::string fileName3 = fileName1 + "-" + std::to_string(1) + ".jpg";
			cv::imwrite(filePath3 + "\\" + fileName3, image_finemapping);

			//Segmentation-based
			if (method == SEGMENTATION_BASED_METHOD)
			{
				image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 2, HorizontalPadding);
				std::string fileName4 = fileName1 + "-" + std::to_string(2) + ".jpg";
				cv::imwrite(filePath4 + "\\" + fileName4, image_finemapping);
				cv::resize(image_finemapping, image_finemapping, cv::Size(136 + HorizontalPadding, 36));
				//cv::imshow("image_finemapping",image_finemapping);
				//cv::waitKey(0);
				plateinfo.setPlateImage(image_finemapping);
				std::vector<cv::Rect> rects;

				plateSegmentation->segmentPlatePipline(plateinfo, 1, rects);
				plateSegmentation->ExtractRegions(plateinfo, rects,fileName1);
				cv::copyMakeBorder(image_finemapping, image_finemapping, 0, 0, 0, 20, cv::BORDER_REPLICATE);
				plateinfo.setPlateImage(image_finemapping);
				generalRecognizer->SegmentBasedSequenceRecognition(plateinfo);
				plateinfo.decodePlateNormal(CH_PLATE_CODE);
			}
			//Segmentation-free
			else if (method == SEGMENTATION_FREE_METHOD)
			{

				image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 4, HorizontalPadding + 3);

				cv::resize(image_finemapping, image_finemapping, cv::Size(136 + HorizontalPadding, 36));
				plateinfo.setPlateImage(image_finemapping);
				std::pair<std::string, float> res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(plateinfo.getPlateImage(), CH_PLATE_CODE);
				plateinfo.confidence = res.second;
				plateinfo.setPlateName(res.first);
			}
			results.push_back(plateinfo);
		}
		return results;
	}
}