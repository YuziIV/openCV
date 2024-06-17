#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>

class dataProcessing
{
public:
    dataProcessing() {}
    ~dataProcessing() {}
    void loadData()
    {
        cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV("./data/emnist_letters_merged.csv", 0, 0, 1); // First col is the target as a float
        samples = tdata->getTrainSamples();                                                                             // Get design matrix  | results in a [784 x 20800] matrix with all features for each letter
        target = tdata->getTrainResponses();                                                                            // Get target values  | results in a [1 x 20800 (800*26 letters)] vector with all the identifications
    }
    void splitData()
    {
        // Training with A and B
        // First 1000 lines for training
        // Last 5000 lines for testing
    }
    void testPrint(cv::Mat &Temp)
    {
        for (int i = 0; i < Temp.rows; ++i)
        {
            //   std::cout << Temp.at<float>(i,0);
        }
        // std::cout << Temp.size();
    }
    void standardizeData(cv::Mat &data)
    {
        mean = cv::Mat::zeros(1, data.cols, CV_32F);
        stdDev = cv::Mat::zeros(1, data.cols, CV_32F);

        cv::Scalar meanScalar, stdDevScalar;
        for (int i = 0; i < data.cols; i++)
        {
            meanStdDev(data.col(i), meanScalar, stdDevScalar);
            // std::cout << "At Pixel:" << i << "Mean:" << meanScalar[0] << "\n";
            mean.at<float>(0, i) = meanScalar[0];
            stdDev.at<float>(0, i) = stdDevScalar[0];
            data.col(i) -= meanScalar[0];
            if (stdDevScalar[0] != 0)
            {
                data.col(i) /= stdDevScalar[0];
            }
        }
    }
    void testPrintMeanStdDev(cv::Mat &Temp)
    {
        cv::Scalar meanScalar, stdDevScalar;
        for (int i = 0; i < Temp.cols; i++)
        {
            meanStdDev(Temp.col(i), meanScalar, stdDevScalar);

            std::cout << "Mittelwert:" << meanScalar[0] << "\n";
            std::cout << "Standardabweichung:" << stdDevScalar[0] << "\n";
        }
    }
    // get the identification vector
    cv::Mat getTarget()
    {
        return target;
    }
    // get the features vector
    cv::Mat getSamples()
    {
        return samples;
    }

private:
    // ensure the format of the data is already provided in .csv Format
    std::string inputDataPath = "./data/emnist_letters_merged.csv";
    cv::Mat samples;
    cv::Mat target;
    cv::Mat mean;
    cv::Mat stdDev;
};

class PCA_
{
public:
private:
};

class SVM
{
public:
private:
};

int main()
{
    dataProcessing dp;
    dp.loadData();
    cv::Mat samples = dp.getSamples();
    dp.standardizeData(samples);
    dp.testPrintMeanStdDev(samples);
    return 0;
};