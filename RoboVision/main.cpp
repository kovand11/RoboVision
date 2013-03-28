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

/*

int main(int argc,char *argv[])
{
	
	

	

	string CurrKey;
	for( int i = 1; i < argc; i++ )
	{
		string Param=string(argv[i]);
		if (Param[0]=='-')
		{
			CurrKey=Param;
		}
		else
		{
			CLP[CurrKey]=Param;
		}
	}

	string inputPath=CLP["-src"];
	string outputPath=CLP["-dst"];
	string task=CLP["-task"];
	string calibPath=CLP["-cal"];
	string detectorName=CLP["-det"];
	string extractorName=CLP["-ext"];
	string matcherName=CLP["-matc"];

	if (task=="featurepoint")
	{
		vector<Mat> inputImages;
		processImageSeq(inputPath,inputImages);
		vector<Mat> outputImages;
		Ptr<FeatureDetector> detector = new SurfFeatureDetector(800);
		drawFeaturePoints(inputImages,outputImages,detector);
		ofstream outputStream(outputPath+"output.txt");
		for (size_t i=0;i<outputImages.size();i++)
		{
			imwrite(outputPath+"img"+to_string(i)+".jpg",outputImages[i]);
			outputStream<<outputPath+"img"+to_string(i)+".jpg"<<endl;
		}
		outputStream.close();

	}
	else if (task=="homography")
	{
		Ptr<FeatureDetector> detector = new SurfFeatureDetector(2500);
		Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor(2500);
		Ptr<DescriptorMatcher> matcher=new FlannBasedMatcher();
		vector<Mat> inputImages;
		processImageSeq(inputPath,inputImages);
		vector<Mat> outputImages;			
		drawHomographySquare(inputImages,outputImages,detector,extractor,matcher,40);
		ofstream outputStream(outputPath+"output.txt");
		for (size_t i=0;i<outputImages.size();i++)
		{
			imwrite(outputPath+"img"+to_string(i)+".jpg",outputImages[i]);
			outputStream<<outputPath+"img"+to_string(i)+".jpg"<<endl;
		}
		outputStream.close();

	}
	else if (task=="3dmodel")
	{
		cout<<"Unimplemented: "<<task<<endl;

		Mat cameraMatrix;
		Mat	distCoeffs;
		Size calibratedImageSize;
		readCameraMatrix(calibPath, cameraMatrix, distCoeffs, calibratedImageSize);
		Ptr<FeatureDetector> detector = FeatureDetector::create(detectorName);
		Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractorName);



	}
	else if (task=="calibrate")
	{
		cout<<"Unimplemented: "<<task<<endl;
	}


}*/





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

void drawHomographySquare(const vector<Mat> &inputImages,vector<Mat> &outputImages,const Ptr<FeatureDetector>& detector,const Ptr<DescriptorExtractor>& extractor,const Ptr<DescriptorMatcher>& matcher,int RefreshRate)
{
	Mat image;
	Mat outputImage;

	vector<KeyPoint> firstKeypoints;
	vector<KeyPoint> keypoints;

	Mat firstDescriptors;
	Mat descriptors;

	vector<DMatch> matches;

	image=inputImages[0].clone();
	cvtColor(image,image,CV_RGB2GRAY);
	detector->detect(image,firstKeypoints);
	extractor->compute(image,firstKeypoints,firstDescriptors);

	for (size_t i=0;i<inputImages.size();i++)
	{		
		image=inputImages[i].clone();
		outputImage=image.clone();
		cvtColor(image,image,CV_RGB2GRAY);
		detector->detect(image,keypoints);
		extractor->compute(image,keypoints,descriptors);

		matcher->match(firstDescriptors,descriptors,matches);
		vector<Point2f> LastPoints;
		vector<Point2f> ThisPoints;
		for (unsigned int j=0;j<matches.size();j++)
		{
			LastPoints.push_back( firstKeypoints[ matches[j].queryIdx ].pt );
			ThisPoints.push_back( keypoints[ matches[j].trainIdx ].pt );
		}
		Mat H=findHomography(LastPoints,ThisPoints,CV_RANSAC);				

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




//-srcdir: <source directory>
//-srcfile: <txt file images newline separated>
//-dst: <destination directory>
//-task: featurepoint homography epipolar 
//-dtxt:SURF ORB


int main(int argc,char *argv[])
{
	map <string,string> CLP; //command line parameters

	CLP["-srcdir"]="testinput/";
	CLP["-srcfile"]="input.txt";
	CLP["-dst"]="testoutput/";
	CLP["-task"]="featurepoint";
	CLP["-dtxt"]="ORB";

	//choosing detector, extractor, and matcher 

	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;

	if (CLP["-dtxt"]=="SURF")
	{
		detector = new SurfFeatureDetector(500);
		extractor = new SurfDescriptorExtractor(500);
		matcher= new BruteForceMatcher< L2<float> >();
	}
	else if (CLP["-dtxt"]=="ORB")
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
		ofstream outputStream(CLP["-dst"]+"output.txt");
		for (size_t i=0;i<outputImages.size();i++)
		{
			//string path=CLP["-dst"]+"img"+to_string(i)+".jpg";
			string path="img"+to_string(i)+".jpg";
			imwrite(path,outputImages[i]);
			outputStream<<CLP["-dst"]+"img"+to_string(i)+".jpg"<<endl;
		}
		outputStream.close();
	}
	else
	{
		vector<Mat> inputImages;
		processImageSeq(CLP["-srcdir"],CLP["-srcfile"],inputImages);
		vector<Mat> outputImages;			
		drawHomographySquare(inputImages,outputImages,detector,extractor,matcher,40);
		ofstream outputStream(CLP["-dst"]+"output.txt");
		for (size_t i=0;i<outputImages.size();i++)
		{
			string path="img"+to_string(i)+".jpg";
			imwrite(path,outputImages[i]);
			outputStream<<CLP["-dst"]+"img"+to_string(i)+".jpg"<<endl;
		}
		outputStream.close();
	}





	return 0;
}