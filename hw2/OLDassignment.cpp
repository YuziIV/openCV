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
    void mergeAndShuffleData(cv::Mat &typeTarget, cv::Mat &typeSample)
    {
        cv::Mat MergedData;
        std::vector<int> indices;
        auto rng = std::default_random_engine{};

        // Merge samples and targets
        cv::hconcat(typeTarget, typeSample, MergedData);

        indices.reserve(MergedData.rows);

        for (int i = 0; i < MergedData.rows; i++)
        {
            indices.push_back(i);
        }

        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), rng);

        cv::Mat shuffledTrainMatrix(MergedData.rows, MergedData.cols, MergedData.type());
        for (long unsigned int i = 0; i < indices.size(); i++)
        {
            MergedData.row(indices[i]).copyTo(shuffledTrainMatrix.row(i));
        }
        MergedData = shuffledTrainMatrix;
        // Split merged data back into samples and targets
        int numCols = trainSamples.cols;
        typeTarget = MergedData.col(0).clone();
        typeSample = MergedData.colRange(0, numCols).clone();
    }

    void splitData()
    {
        // Training with A and B
        // 1000             (2 letters * 500 variations each) lines for training
        // Unsused 5018     (26 lettes * 193 variations each) lines for testing
        for (int i = 0; i < 1600; i += 800)
        {
            // Split samples
            trainSamples.push_back(samples.rowRange(i, i + 500));
            testSamples.push_back(samples.rowRange(i + 500, i + 500 + 193));

            // Split target
            trainTarget.push_back(target.rowRange(i, i + 500));
            testTarget.push_back(target.rowRange(i + 500, i + 500 + 193));
        }
        for (int i = 1600; i < 20800; i += 800)
        {

            testSamples.push_back(samples.rowRange(i, i + 193));

            testTarget.push_back(target.rowRange(i, i + 193));
        }
    }

    void testPrint(cv::Mat &data)
    {
        for (int i = 0; i < data.rows; ++i)
        {
            std::cout << data.at<float>(i, 68) << " ";
        }
        std::cout << data.size() << std::endl;
    }

    // use = 1 if you want to apply the calculated Standartizations parameters on another set
    void standardizeData(cv::Mat &data, bool use) // standardize
    {
        // Initialize mean and standard deviation matrices only if 'use' is false
        if (!use)
        {
            mean = cv::Mat::zeros(1, data.cols, CV_32F);   // initiate the matrices
            stdDev = cv::Mat::zeros(1, data.cols, CV_32F); // initiate the matrices
        }

        for (int i = 0; i < data.cols; i++)
        {
            if (!use)
            {
                // Calculate mean and standard deviation for each column/Pixel
                meanStdDev(data.col(i), meanScalar, stdDevScalar);
                mean.at<float>(0, i) = meanScalar[0];
                stdDev.at<float>(0, i) = stdDevScalar[0];
            }

            // Standardize the data
            data.col(i) -= mean.at<float>(0, i);
            if (stdDev.at<float>(0, i) != 0)
            {
                data.col(i) /= stdDev.at<float>(0, i);
            }
        }
    }

    void testPrintMeanStdDev(cv::Mat &data, bool wholeMat)
    {
        if (wholeMat)
        {
            meanStdDev(data, meanScalar, stdDevScalar);
            std::cout << "Mittelwert:" << meanScalar[0] << "\n";
            std::cout << "Standardabweichung:" << stdDevScalar[0] << "\n\n";
        }
        else
        {
            for (int i = 0; i < data.cols; i++)
            {
                meanStdDev(data.col(i), meanScalar, stdDevScalar);

                std::cout << "Mittelwert:" << meanScalar[0] << "\n";
                std::cout << "Standardabweichung:" << stdDevScalar[0] << "\n\n";
            }
        }
    }

    /* get the identification vectors
    1 for Training
    2 for Testing
    0 / default for All*/
    cv::Mat &getTarget(int tmp)
    {
        switch (tmp)
        {
        case 1:
            // std::cout << "Train Target data is returned!\n";
            return trainTarget;
        case 2:
            // std::cout << "Test Target data is returned!\n";
            return testTarget;
        default:
            // std::cout << "Whole Target data is returned!\n";
            return target;
        }
    }

    /* get the features vectors
    1 for Training
    2 for Testing
    0 / default for All*/
    cv::Mat &getSamples(int tmp)
    {
        switch (tmp)
        {
        case 1:
            // std::cout << "Train Sample data is returned!\n";
            return trainSamples;
        case 2:
            // std::cout << "Test Sample data is returned!\n";
            return testSamples;
        default:
            // std::cout << "Whole Sample data is returned!\n";
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

    cv::Scalar meanScalar, stdDevScalar;

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
    dp.splitData();
    dp.mergeAndShuffleData(dp.getTarget(1), dp.getSamples(1)); // shuffle Trainingsdata
    dp.mergeAndShuffleData(dp.getTarget(2), dp.getSamples(2)); // shuffle Testdata
    dp.mergeAndShuffleData(dp.getTarget(0), dp.getSamples(0)); // shuffle Testdata

    cv::Mat Stichprobe = dp.getSamples(0).clone();
    dp.standardizeData(Stichprobe, 0);

    cv::Mat trainingSamples = dp.getSamples(1);
    cv::Mat testingSamples = dp.getSamples(2);
    // dp.testPrint(samples);
    dp.standardizeData(trainingSamples, 1);
    dp.testPrintMeanStdDev(trainingSamples, 0);

    dp.standardizeData(testingSamples, 1);
    dp.testPrintMeanStdDev(testingSamples, 1);

    return 0;
};