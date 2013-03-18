#include"robovision.h"

#include<iostream>
#include<fstream>

using namespace std;
using namespace cv;


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

void drawFeaturePoints(VideoCapture &Input,VideoWriter &Output,const Ptr<FeatureDetector>& detector)
{
	int FrameCount=static_cast<int>(Input.get(CV_CAP_PROP_FRAME_COUNT));
	for (int i=0;i<FrameCount;i++)
	{
		Mat TMPImage;
		Mat OutImage;
		vector<KeyPoint> TMPKeypoints;
		Input>>TMPImage;
		OutImage=TMPImage.clone();
		cvtColor(TMPImage,TMPImage,CV_RGB2GRAY);
		detector->detect(TMPImage,TMPKeypoints);
		drawKeypoints(OutImage,TMPKeypoints,OutImage,cv::Scalar(255,255,255));
		Output<<OutImage;
		cout<<i+1<<"/"<<FrameCount<<endl;
	}
	return;
}

void drawHomographySquare(VideoCapture &Input,VideoWriter &Output,const Ptr<FeatureDetector>& detector,const Ptr<DescriptorExtractor>& extractor,const Ptr<DescriptorMatcher>& matcher,int RefreshRate)
{
	int FrameCount=static_cast<int>(Input.get(CV_CAP_PROP_FRAME_COUNT));

	Mat ThisImage;
	Mat LastImage;
	Mat OutputImage;

	Mat ThisDescriptors;
	Mat LastDescriptors;

	vector<KeyPoint> ThisKeypoints;
	vector<KeyPoint> LastKeypoints;

	vector<DMatch> Matches;

	for (int i=0;i<FrameCount;i++)
	{		
		cout<<i+1<<"/"<<FrameCount<<endl;
		Input>>ThisImage;
		OutputImage=ThisImage.clone();
		cvtColor(ThisImage,ThisImage,CV_RGB2GRAY);
		detector->detect(ThisImage,ThisKeypoints);
		extractor->compute(ThisImage,ThisKeypoints,ThisDescriptors);

		if (i%RefreshRate==0)
		{
			LastImage=ThisImage.clone();
			LastKeypoints=ThisKeypoints;
			LastDescriptors=ThisDescriptors.clone();
		}		

		if (LastKeypoints.size()>=8 && ThisKeypoints.size()>=8)
		{
			matcher->match(LastDescriptors,ThisDescriptors,Matches);
			vector<Point2f> LastPoints;
			vector<Point2f> ThisPoints;
			for (unsigned int j=0;j<Matches.size();j++)
			{
				LastPoints.push_back( LastKeypoints[ Matches[j].queryIdx ].pt );
				ThisPoints.push_back( ThisKeypoints[ Matches[j].trainIdx ].pt );
			}
			Mat H=findHomography(LastPoints,ThisPoints,CV_RANSAC);				

			int Cols=OutputImage.cols;
			int Rows=OutputImage.rows;
		

			std::vector<Point2f> BaseCorners(4);
			BaseCorners[0] = cvPoint(Cols/4,Rows/4);
			BaseCorners[1] = cvPoint((Cols*3/4), Rows/4 );
			BaseCorners[2] = cvPoint( (Cols*3)/4, (Rows*3)/4 );
			BaseCorners[3] = cvPoint( Cols/4, (Rows*3)/4 );

			std::vector<Point2f> TransCorners(4);
			perspectiveTransform( BaseCorners, TransCorners,H);

			line(OutputImage, BaseCorners[0], BaseCorners[1], Scalar(0, 255, 0), 1 );
			line(OutputImage, BaseCorners[1] , BaseCorners[2], Scalar( 0, 255, 0), 1 );
			line(OutputImage, BaseCorners[2], BaseCorners[3], Scalar( 0, 255, 0), 1 );
			line(OutputImage, BaseCorners[3], BaseCorners[0], Scalar( 0, 255, 0), 1 );

			line(OutputImage, TransCorners[0], TransCorners[1], Scalar(0, 0, 255), 1 );
			line(OutputImage, TransCorners[1], TransCorners[2], Scalar( 0, 0, 255), 1 );
			line(OutputImage, TransCorners[2], TransCorners[3], Scalar( 0, 0, 255), 1 );
			line(OutputImage, TransCorners[3], TransCorners[0], Scalar( 0, 0, 255), 1 );
			

			Output<<OutputImage;			
		}
	}




	return;
}

void build3dModel()
{

}

int main(int argc,char *argv[])
{
	
	map <string,string> CLP;

	CLP["-src"]="D:\\drone.mp4"; //input file
	CLP["-dst"]="D:\\output.avi";//output file
	CLP["-task"]="featurepoint";//task to do (featurepoint homography 3dmodel calibrate)
	CLP["cal"]="cal.yml";//calibration file
	CLP["-det"]="SURF"; //detector
	CLP["-ext"]="SURF"; //extractor
	CLP["-matc"]="FlannBased"; //matcher

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
		Ptr<FeatureDetector> detector = FeatureDetector::create(detectorName);
		VideoCapture inputVideo(inputPath);
		int Width=static_cast<int>(inputVideo.get(CV_CAP_PROP_FRAME_WIDTH));
		int Height=static_cast<int>(inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));			
		VideoWriter outputVideo( outputPath ,CV_FOURCC('D', 'I', 'V', 'X'),inputVideo.get(CV_CAP_PROP_FPS),Size(Width,Height));
		drawFeaturePoints(inputVideo,outputVideo,detector);

	}
	else if (task=="homography")
	{
		Ptr<FeatureDetector> detector = FeatureDetector::create(detectorName);
		Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractorName);
		Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create(matcherName);
		VideoCapture inputVideo(inputPath);
		int Width=static_cast<int>(inputVideo.get(CV_CAP_PROP_FRAME_WIDTH));
		int Height=static_cast<int>(inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));			
		VideoWriter outputVideo( outputPath ,CV_FOURCC('D', 'I', 'V', 'X'),inputVideo.get(CV_CAP_PROP_FPS),Size(Width,Height));
		drawHomographySquare(inputVideo,outputVideo,detector,extractor,matcher,40);

	}
	else if (task=="3dmodel")
	{
		cout<<"Unimplemented: "<<task<<endl;

	}
	else if (task=="calibrate")
	{
		cout<<"Unimplemented: "<<task<<endl;
	}


}



