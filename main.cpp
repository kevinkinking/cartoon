#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>

using namespace std;

const size_t inWidth = 800;
const size_t inHeight = 800;
const float inScaleFactor = 0.007843f;//0.007843f
const float meanVal_R = 127.5f;
const float meanVal_G = 127.5f;
const float meanVal_B = 127.5f;
const float confidenceThreshold = 0.50f;

string class_names[39] = {"bg","girl_skrztj", "girl_sxqb", "girl_dsws", "girl_wczy", "girl_tzbdh", "girl_shylscls", "girl_ydkws", 
"girl_zcjg", "girl_sswj", "girl_ahws", "girl_zjzl", "girl_zjxs", "girl_zjxl", "girl_zjcf", 
"girl_asqc", "girl_lyzr", "girl_zjsj", "girl_ddfx", "girl_skyjj", "girl_blfpq", 
"girl_xhjpy", "girl_assx", "girl_skdsswdn", "girl_zjzlsb", "girl_wmlm", "girl_zjdszjz", "girl_dhs", 
"girl_swsj", "girl_dlst", "girl_rzxx", "girl_lxgq", "girl_jjyg", "girl_dmmdxbs", "girl_zjcy", 
"girl_rald", "girl_dcsgsc", "girl_qxsqxz", "xx",};
vector<string> img_names = {
                            "test.jpg",
                            "30.jpg",
                            "45.jpg",
                            "60.jpg",
                            "75.jpg",
                            "90.jpg",
                            "105.jpg",
                            };
dnn::Net net;

bool ini_model(string txt_file,string bin_file)
{
    net = readNetFromCaffe(txt_file, bin_file);
    cout<<"------use opencl-------"<<endl;
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    if (net.empty())
    {
        //这句话没有打印就代表模型初始化成功
        cerr << "Can't load network by using the input files: " << endl;
        exit(-1);//报错退出
    }
    return true;
}

void detect_img(cv::Mat frame,vector<cv::Rect> &rects,vector<int> &ids)
{
    if (frame.empty())
    {
        return;
    }

    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);

    Mat inputBlob = blobFromImage(frame, inScaleFactor, Size(inWidth, inHeight), Scalar(meanVal_R, meanVal_G, meanVal_B), false, false);

    net.setInput(inputBlob, "data");

    Mat detection = net.forward("detection_out");


    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    cout<<detection.size<<endl;


    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        cout<<confidence<<endl;
        if(confidence > confidenceThreshold)
        {
            float objectClass = detectionMat.at<float>(i, 1);
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
            rects.push_back(cv::Rect(xLeftBottom,yLeftBottom,xRightTop-xLeftBottom,yRightTop-yLeftBottom));

            ids.push_back(int(objectClass));
        }
    }
}

int main(int argc, char** argv)
{
    ini_model("../model/cartoon0631/cartoon.txt", "../model/cartoon0631/cartoon.bin");

    vector<string>::iterator img_name_iter;
    for(img_name_iter = img_names.begin();img_name_iter!=img_names.end();img_name_iter++)
    {
        string img_path = "../imgs/" + *img_name_iter;
        cout<<img_path<<endl;
        cv::Mat img = imread(img_path);

        resize(img, img, Size(inWidth, inHeight));
        vector<cv::Rect> rects;
        vector<int> ids;

        double time_b = getTickCount();
        detect_img(img,rects,ids);
        double time_e = getTickCount();
        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;

        double time = (time_e - time_b) / freq;
        cout << "Inference time, ms: " << time << endl;

        for(int i=0;i<rects.size();i++)
        {
            rectangle(img,rects[i],Scalar(0,255,0));
            if(ids[i] == 11){
                continue;
            }
            putText(img,class_names[ids[i]],Point(rects[i].x+5,rects[i].y+20),1,1,Scalar(0,0,255));
        }
        imshow("ssd",img);
        int key = waitKey(0);
        if(key==27)
        {
            break;
        }
    }

    return 0;
}
