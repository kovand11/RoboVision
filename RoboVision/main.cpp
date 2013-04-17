#include"robovision.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>

#include<iostream>
#include<fstream>

using namespace std;
using namespace cv;







void drawFeaturePoints(const vector<Mat> &inputImages,vector<Mat> &outputImages,const Ptr<FeatureDetector>& detector)
{
	for (size_t i=0;i<inputImages.size();i++)
	{
		Mat tmpImage;
		Mat outImage;
		vector<KeyPoint> keypoints;
		tmpImage=inputImages[i].clone();
		outImage=tmpImage.clone();
		cvtColor(tmpImage,tmpImage,CV_RGB2GRAY);
		detector->detect(tmpImage,keypoints);
		drawKeypoints(outImage,keypoints,outImage,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		outputImages.push_back(outImage);
		cout<<i+1<<"/"<<inputImages.size()<<endl;
	}
	return;
}

void readCameraMatrix(const string &fileName,Mat &cameraMatrix, Mat &distCoeffs,Size &calibratedImageSize )
{
    FileStorage fs(fileName, FileStorage::READ);
    fs["image_width"] >> calibratedImageSize.width;
    fs["image_height"] >> calibratedImageSize.height;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;

    if( distCoeffs.type() != CV_64F )
        distCoeffs = Mat_<double>(distCoeffs);
    if( cameraMatrix.type() != CV_64F )
        cameraMatrix = Mat_<double>(cameraMatrix);
}

void processImageSeq(const string &path,const string &inputfile,vector<Mat> &images)
 {
	 ifstream inputStream(path+inputfile);
	while(!inputStream.eof())
	{
		string line;
		inputStream>>line;
		images.push_back(imread(path+line));
	}
	inputStream.close();

}

void drawHomographySquare(vector<Mat> &inputImages,vector<Mat> &outputImages,Ptr<FeatureDetector> detector,Ptr<DescriptorExtractor> extractor,const Ptr<DescriptorMatcher>& matcher,int keyframe)
{
	Mat outputImage;
	vector< vector<KeyPoint> > keypointsVec;
	vector<Mat> descriptorsVec;

	detector->detect(inputImages,keypointsVec);
	extractor->compute(inputImages,keypointsVec,descriptorsVec);	

	vector< vector<DMatch> > forwardMatches2;
	vector< vector<DMatch> > backwardMatches2;
	vector<DMatch> forwardMatches;
	vector<DMatch> backwardMatches;
	vector<DMatch> matches;

	smartMatcher sMatcher=smartMatcher();


	for (size_t i=0;i<inputImages.size();i++)
	{
		sMatcher.symmetryMatch2(descriptorsVec[i],descriptorsVec[keyframe],forwardMatches2,backwardMatches2,matcher);
		sMatcher.ratioTest2(forwardMatches2,forwardMatches,0.65);
		sMatcher.ratioTest2(backwardMatches2,backwardMatches,0.65);
		sMatcher.symmetryTest(forwardMatches,backwardMatches,matches);
		sMatcher.ransacTest(matches,keypointsVec[i],keypointsVec[keyframe],matches,3.0,0.98);

		vector<Point2f> points;
		vector<Point2f> keyPoints;

		for (size_t j=0;j<matches.size();j++)
		{
			points.push_back( keypointsVec[i][ matches[j].queryIdx ].pt );
			keyPoints.push_back( keypointsVec[keyframe][ matches[j].trainIdx ].pt );
		}
		Mat H=findHomography(keyPoints,points,CV_RANSAC,2.5);

		outputImage=inputImages[i].clone();
		int cols=outputImage.cols;
		int rows=outputImage.rows;

		std::vector<Point2f> BaseCorners(4);
		BaseCorners[0] = cvPoint(cols/4,rows/4);
		BaseCorners[1] = cvPoint((cols*3/4), rows/4 );
		BaseCorners[2] = cvPoint( (cols*3)/4, (rows*3)/4 );
		BaseCorners[3] = cvPoint( cols/4, (rows*3)/4 );

		std::vector<Point2f> TransCorners(4);
		perspectiveTransform( BaseCorners, TransCorners,H);

		line(outputImage, BaseCorners[0], BaseCorners[1], Scalar(0, 255, 0), 1 );
		line(outputImage, BaseCorners[1] , BaseCorners[2], Scalar( 0, 255, 0), 1 );
		line(outputImage, BaseCorners[2], BaseCorners[3], Scalar( 0, 255, 0), 1 );
		line(outputImage, BaseCorners[3], BaseCorners[0], Scalar( 0, 255, 0), 1 );

		line(outputImage, TransCorners[0], TransCorners[1], Scalar(0, 0, 255), 1 );
		line(outputImage, TransCorners[1], TransCorners[2], Scalar( 0, 0, 255), 1 );
		line(outputImage, TransCorners[2], TransCorners[3], Scalar( 0, 0, 255), 1 );
		line(outputImage, TransCorners[3], TransCorners[0], Scalar( 0, 0, 255), 1 );



		outputImages.push_back(outputImage);
	}
	return;
}

void drawGoodMatches(vector<Mat> &inputImages,vector<Mat> &outputImages,Ptr<FeatureDetector> detector,Ptr<DescriptorExtractor> extractor,const Ptr<DescriptorMatcher>& matcher,int keyframe)
{
	Mat outputImage;
	vector< vector<KeyPoint> > keypointsVec;
	vector<Mat> descriptorsVec;

	detector->detect(inputImages,keypointsVec);
	extractor->compute(inputImages,keypointsVec,descriptorsVec);	

	vector< vector<DMatch> > forwardMatches2;
	vector< vector<DMatch> > backwardMatches2;
	vector<DMatch> forwardMatches;
	vector<DMatch> backwardMatches;
	vector<DMatch> matches;

	smartMatcher sMatcher=smartMatcher();


	for (size_t i=0;i<inputImages.size();i++)
	{
		sMatcher.symmetryMatch2(descriptorsVec[i],descriptorsVec[keyframe],forwardMatches2,backwardMatches2,matcher);
		sMatcher.ratioTest2(forwardMatches2,forwardMatches,0.85);
		sMatcher.ratioTest2(backwardMatches2,backwardMatches,0.85);
		sMatcher.symmetryTest(forwardMatches,backwardMatches,matches);
		sMatcher.ransacTest(matches,keypointsVec[i],keypointsVec[keyframe],matches,3.0,0.99);

		vector<Point2f> points;
		vector<Point2f> keyPoints;

		for (size_t j=0;j<matches.size();j++)
		{
			points.push_back( keypointsVec[i][ matches[j].queryIdx ].pt );
			keyPoints.push_back( keypointsVec[keyframe][ matches[j].trainIdx ].pt );
		}
		Mat H=findHomography(keyPoints,points,CV_RANSAC,2.5);

		outputImage=inputImages[i].clone();

		drawMatches(inputImages[i],keypointsVec[i],inputImages[keyframe],keypointsVec[keyframe],matches,outputImage);

		outputImages.push_back(outputImage);
	}
	return;
}




//-srcdir: <source directory>
//-srcfile: <txt file images newline separated>
//-dst: <destination directory>
//-task: featurepoint homography epipolar 3drecon
//-dtxt:SURF ORB
//-keyfr: <keyframe>




int main(int argc,char *argv[])
{
	map <string,string> CLP; //command line parameters

	CLP["-srcdir"]="testinput/";
	CLP["-srcfile"]="input.txt";
	CLP["-dst"]="testoutput/";
	CLP["-task"]="featurepoint";
	CLP["-dtxt"]="ORB";

	string currentFlag;
	for( int i = 1; i < argc; i++ )
	{
		string Param=string(argv[i]);
		if (Param[0]=='-')
		{
			currentFlag=Param;
		}
		else
		{
			CLP[currentFlag]=Param;
		}
	}

	//choosing detector, extractor, and matcher 

	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;

	if (CLP["-dtxt"]=="surf")
	{
		detector = new SurfFeatureDetector();
		extractor = new SurfDescriptorExtractor();
		matcher= new BruteForceMatcher< L2<float> >();
	}
	else if (CLP["-dtxt"]=="orb")
	{
		detector = new OrbFeatureDetector();
		extractor = new OrbDescriptorExtractor();
		matcher= new BruteForceMatcher<Hamming>();
	}
	else
	{
		cout<<"No such dtxt: "<<CLP["-dtxt"]<<endl;
	}

	if (CLP["-task"]=="featurepoint")
	{
		vector<Mat> inputImages;
		processImageSeq(CLP["-srcdir"],CLP["-srcfile"],inputImages);
		vector<Mat> outputImages;
		drawFeaturePoints(inputImages,outputImages,detector);
		ofstream outputStream(CLP["-dst"]+"log.txt");
		for (size_t i=0;i<outputImages.size();i++)
		{
			//string path=CLP["-dst"]+"img"+to_string(i)+".jpg";
			string path=CLP["-dst"]+"img"+to_string(i)+".jpg";
			imwrite(path,outputImages[i]);
			outputStream<<CLP["-dst"]+"img"+to_string(i)+".jpg"<<endl;
		}
		outputStream.close();
	}
	else if (CLP["-task"]=="homography")
	{
		vector<Mat> inputImages;
		processImageSeq(CLP["-srcdir"],CLP["-srcfile"],inputImages);
		vector<Mat> outputImages;			
		drawHomographySquare(inputImages,outputImages,detector,extractor,matcher,atoi(CLP["-keyfr"].c_str()));
		ofstream outputStream(CLP["-dst"]+"log.txt");
		for (size_t i=0;i<outputImages.size();i++)
		{
			string path=CLP["-dst"]+"img"+to_string(i)+".jpg";
			imwrite(path,outputImages[i]);
			outputStream<<CLP["-dst"]+"img"+to_string(i)+".jpg"<<endl;
		}
		outputStream.close();
	}
	else if (CLP["-task"]=="matches")
	{
		vector<Mat> inputImages;
		processImageSeq(CLP["-srcdir"],CLP["-srcfile"],inputImages);
		vector<Mat> outputImages;			
		drawGoodMatches(inputImages,outputImages,detector,extractor,matcher,10);
		ofstream outputStream(CLP["-dst"]+"log.txt");
		for (size_t i=0;i<outputImages.size();i++)
		{
			string path=CLP["-dst"]+"img"+to_string(i)+".jpg";
			imwrite(path,outputImages[i]);
			outputStream<<CLP["-dst"]+"img"+to_string(i)+".jpg"<<endl;
		}
		outputStream.close();
	}

	return 0;
}



