#include"robovision.h"

#include<iostream>
#include<fstream>

using namespace std;
using namespace cv;


void DrawFeaturePoints(VideoCapture &Input,VideoWriter &Output,FeatureDetector &Detector)
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
		Detector.detect(TMPImage,TMPKeypoints);
		drawKeypoints(OutImage,TMPKeypoints,OutImage,cv::Scalar(255,255,255));
		Output<<OutImage;
		cout<<i+1<<"/"<<FrameCount<<endl;
	}
	return;
}

void DrawHomographySquare(VideoCapture &Input,VideoWriter &Output,FeatureDetector &Detector,DescriptorExtractor &Extractor,DescriptorMatcher &Matcher,int RefreshRate)
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
		Detector.detect(ThisImage,ThisKeypoints);
		Extractor.compute(ThisImage,ThisKeypoints,ThisDescriptors);

		if (i%RefreshRate==0)
		{
			LastImage=ThisImage.clone();
			LastKeypoints=ThisKeypoints;
			LastDescriptors=ThisDescriptors.clone();
		}		

		if (LastKeypoints.size()>=8 && ThisKeypoints.size()>=8)
		{
			Matcher.match(LastDescriptors,ThisDescriptors,Matches);
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
		else
		{
		}
	}




	return;
}

int main(int argc,char *argv[])
{

	SurfFeatureDetector Detector(800);
	SurfDescriptorExtractor Extractor(800);
	FlannBasedMatcher Matcher;

	map <string,string> CLP;

	CLP["-src"]="D:\\drone.mp4";
	CLP["-dst"]="D:\\output.avi";
	CLP["-task"]="featurepoint";


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

	string InputPath=CLP["-src"];
	string OutputPath=CLP["-dst"];
	string Task=CLP["-task"];



	if (Task=="featurepoint")
	{
		VideoCapture InputVideo(InputPath);
		int Width=static_cast<int>(InputVideo.get(CV_CAP_PROP_FRAME_WIDTH));
		int Height=static_cast<int>(InputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));			
		VideoWriter OutputVideo( OutputPath ,CV_FOURCC('D', 'I', 'V', 'X'),InputVideo.get(CV_CAP_PROP_FPS),Size(Width,Height));
		DrawFeaturePoints(InputVideo,OutputVideo,Detector);

	}
	else if (Task=="homography")
	{
		VideoCapture InputVideo(InputPath);
		int Width=static_cast<int>(InputVideo.get(CV_CAP_PROP_FRAME_WIDTH));
		int Height=static_cast<int>(InputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));			
		VideoWriter OutputVideo( OutputPath ,CV_FOURCC('D', 'I', 'V', 'X'),InputVideo.get(CV_CAP_PROP_FPS),Size(Width,Height));
		DrawHomographySquare(InputVideo,OutputVideo,Detector,Extractor,Matcher,40);

	}


}



