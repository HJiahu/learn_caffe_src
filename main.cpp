#include<iostream>
#include<vector>
#include"classifier.h"

using namespace std;

int main()
{
    Classifier classifier ("D:/age_gender/caffenet/age");
    vector<Prediction> results;
    cv::Mat img = cv::imread ("D:/age_gender/face_100.JPG");
    
    if (img.empty())
    {
        cout << "can not open this file img" << endl;
        getchar();
    }
    
    results = classifier.Classify (img);

#ifdef _MSC_VER
	system("pause");
#endif

}