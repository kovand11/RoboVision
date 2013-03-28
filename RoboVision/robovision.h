#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>

#include<algorithm>

using namespace std;
using namespace cv;

class RobustMatcher
{
public:
	RobustMatcher(const Ptr<DescriptorMatcher>& matcher,int matchDepht,bool crossCheck)
	{
		_matcher=matcher;
		_toCrossCheck=crossCheck;
	}
	void match(const Mat & descriptor1,const Mat &descriptor2,vector<DMatch> robustMatches)
	{

	}
protected:
	Ptr<DescriptorMatcher> _matcher;
	bool _toCrossCheck;
};
