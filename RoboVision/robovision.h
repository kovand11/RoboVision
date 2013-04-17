#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>

#include<iostream>

#include<algorithm>

using namespace std;
using namespace cv;



class smartMatcher
{

public:

	smartMatcher() 
	{
		
	}

	void symmetryMatch2(Mat descriptors1,Mat descriptors2,vector< vector<DMatch> > &forward2Matches,vector< vector<DMatch> > &backward2Matches,const Ptr<DescriptorMatcher>& matcher)
	{

		matcher->knnMatch(descriptors1,descriptors2,forward2Matches,2);
		matcher->knnMatch(descriptors2,descriptors1,backward2Matches,2);
	}

	// Clear matches for which NN ratio is > than threshold or not having 2 neighbours
	void ratioTest2(vector< vector<DMatch> > &matches2,vector<DMatch> &goodMatches,double ratio=0.85) 
	{
		goodMatches.clear();
		for (vector<vector<cv::DMatch>>::iterator matchIt= matches2.begin();matchIt!= matches2.end(); matchIt++) 
		{
			if (matchIt->size() > 1)
			{
				if ((*matchIt)[0].distance/(*matchIt)[1].distance < ratio)
				{
					goodMatches.push_back((*matchIt)[0]);
				}
			} 
		}
	}


	// Insert symmetrical matches in symMatches vector
	void symmetryTest(const std::vector<cv::DMatch> &matches1,const std::vector<cv::DMatch> &matches2,std::vector<cv::DMatch>& symMatches)
	{
		symMatches.clear();
		for (vector<DMatch>::const_iterator matchIterator1= matches1.begin();matchIterator1!= matches1.end(); ++matchIterator1)
		{
			for (vector<DMatch>::const_iterator matchIterator2= matches2.begin();matchIterator2!= matches2.end();++matchIterator2)
			{
				if ((*matchIterator1).queryIdx ==(*matchIterator2).trainIdx &&(*matchIterator2).queryIdx ==(*matchIterator1).trainIdx)
				{
					symMatches.push_back(DMatch((*matchIterator1).queryIdx,(*matchIterator1).trainIdx,(*matchIterator1).distance));
					break;
				}
			}
		}
	}

	// Identify good matches using RANSAC
	// Return fundemental matrix
	void ransacTest(const std::vector<cv::DMatch> matches,const std::vector<cv::KeyPoint>& keypoints1,const std::vector<cv::KeyPoint>& keypoints2,	std::vector<cv::DMatch>& goodMatches,double distance,double confidence)
	{
		goodMatches.clear();
		// Convert keypoints into Point2f
		std::vector<cv::Point2f> points1, points2;
		for (std::vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it)
		{
			// Get the position of left keypoints
			float x= keypoints1[it->queryIdx].pt.x;
			float y= keypoints1[it->queryIdx].pt.y;
			points1.push_back(cv::Point2f(x,y));
			// Get the position of right keypoints
			x= keypoints2[it->trainIdx].pt.x;
			y= keypoints2[it->trainIdx].pt.y;
			points2.push_back(cv::Point2f(x,y));
		}
		// Compute F matrix using RANSAC
		std::vector<uchar> inliers(points1.size(),0);
		cv::Mat fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2),inliers,CV_FM_RANSAC,distance,confidence); // confidence probability
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator
		itIn= inliers.begin();
		std::vector<cv::DMatch>::const_iterator
		itM= matches.begin();
		// for all matches
		for ( ;itIn!= inliers.end(); ++itIn, ++itM)
		{
			if (*itIn)
			{ // it is a valid match
				goodMatches.push_back(*itM);
			}
		}
	}
};