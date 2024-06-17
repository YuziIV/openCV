#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

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

    void splitData(cv::Mat &data, bool shuffle)
    {
        // Training with A and B
        // 1014 (26 letters * 39 variations) lines for training
        // Unsused 5000 lines for testing
        for (int i = 0; i < 20800; i += 800)
        {
            // Split samples
            trainSamples.push_back(samples.rowRange(i, i + 39));
            testSamples.push_back(samples.rowRange(i + 39, i + 800 - 39));

            // Split target
            trainTarget.push_back(target.rowRange(i, i + 39));
            testTarget.push_back(target.rowRange(i + 39, i + 800 - 39));
        }
        
        if (shuffle)
        {

        }
    }

    void testPrint(cv::Mat &data)
    {
        for (int i = 0; i < data.rows; ++i)
        {
            std::cout << data.at<float>(i, 0);
        }
        std::cout << data.size() << std::endl;
    }

    void standardizeData(cv::Mat &data) // standardize
    {
        mean = cv::Mat::zeros(1, data.cols, CV_32F);   // initiating the matricies
        stdDev = cv::Mat::zeros(1, data.cols, CV_32F); // initiating the matricies

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

    void testPrintMeanStdDev(cv::Mat &data)
    {
        cv::Scalar meanScalar, stdDevScalar;
        for (int i = 0; i < data.cols; i++)
        {
            meanStdDev(data.col(i), meanScalar, stdDevScalar);

            std::cout << "Mittelwert:" << meanScalar[0] << "\n";
            std::cout << "Standardabweichung:" << stdDevScalar[0] << "\n\n";
        }
    }

    /* get the identification vectors
    1 for Training
    2 for Testing
    0 / default for All*/
    cv::Mat getTarget(int tmp)
    {
        switch (tmp)
        {
        case 1:
            std::cout << "Train Target data is returned!\n";
            return trainTarget;
        case 2:
            std::cout << "Test Target data is returned!\n";
            return testTarget;
        default:
            std::cout << "Whole Target data is returned!\n";
            return target;
        }
    }

    /* get the features vectors
    1 for Training
    2 for Testing
    0 / default for All*/
    cv::Mat getSamples(int tmp)
    {
        switch (tmp)
        {
        case 1:
            std::cout << "Train Sample data is returned!\n";
            return trainSamples;
        case 2:
            std::cout << "Test Sample data is returned!\n";
            return testSamples;
        default:
            std::cout << "Whole Sample data is returned!\n";
            return samples;
        }
    }

private:
    // ensure the format of the data is already provided in .csv Format
    std::string inputDataPath = "./data/emnist_letters_merged.csv";
    cv::Mat samples;
    cv::Mat target;

    cv::Mat mean;
    cv::Mat stdDev;

    cv::Mat trainSamples;
    cv::Mat testSamples;
    cv::Mat testTarget;
    cv::Mat trainTarget;

    // int numTrainingSamples = 1014; // (26 letters * 39 variations)
    // int numTestSamples = 5000;
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
    cv::Mat samples = dp.getSamples(0);
    // dp.standardizeData(samples);
    //  dp.testPrintMeanStdDev(samples);
    dp.splitData(samples, 0);
    cv::Mat training = dp.getSamples(1);
    // dp.testPrint(training);
    return 0;
};