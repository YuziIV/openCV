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

    void mergeAndShuffleData()
    {
        cv::Mat MergedData;
        std::vector<int> indices;
        auto rng = std::default_random_engine{};

        // Merge samples and targets
        cv::hconcat(target, samples, MergedData);

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
        int numCols = MergedData.cols;
        target = MergedData.col(0).clone();
        samples = MergedData.colRange(1, numCols).clone();
    }

    // void sampleAndFilter()
    // {
    //     // Training with A and B
    //     // First 1000 lines for training
    //     // Following 5000 lines for testing
    //     for (int i = 0; i < 8000; i++)
    //     {

    //         standatisationSet.push_back(samples.row(i)); // FIX not ideal

    //         float currentLetter = target.at<float>(0, i);
    //         if (currentLetter == 1 || currentLetter == 2)
    //         {
    //             // collecting sample for training
    //             trainSamples.push_back(samples.row(i));
    //             trainTarget.push_back(target.at<float>(0, i));
    //             // std::cout << samples.row(i) << std::endl;
    //         }
    //     }
    //     // uncomment if Training set needs to be only A and B
    //     // for (int i = 1000; i < 6000; i++)
    //     // {
    //     //     if (target.at<float>(0, i) == 1 || 2)
    //     //     {
    //     //         // collecting sample for testing
    //     //         trainSamples.push_back(samples.at<float>(0, i));
    //     //         trainTarget.push_back(target.at<float>(0, i));
    //     //     }
    //     // }
    //     // collecting sample for testing
    //     testSamples.push_back(samples.rowRange(8000, 13000));
    //     testTarget.push_back(target.rowRange(8000, 13000));

    //     standardizeData(standatisationSet, 0); // FIX not ideal
    // }
    void sampleAndFilter()
    {
        // Assume data is already shuffled and loaded into 'samples' and 'target'.
        // Now, filter out the samples corresponding to the labels A (1) and B (2).

        cv::Mat mask = (target == 1) | (target == 2); // Create a mask for rows with A or B
        cv::Mat filteredSamples, filteredTargets;
        for (int i = 0; i < mask.rows; i++)
        {
            if (mask.at<bool>(i, 0)) // Use the mask to filter
            {
                filteredSamples.push_back(samples.row(i));
                filteredTargets.push_back(target.row(i));
            }
        }

        // Split the data into training and testing
        int trainingSize = std::round(filteredSamples.rows * 0.625); // 80% for training
        trainSamples = filteredSamples.rowRange(0, trainingSize);
        trainTarget = filteredTargets.rowRange(0, trainingSize);

        testSamples = filteredSamples.rowRange(trainingSize, filteredSamples.rows);
        testTarget = filteredTargets.rowRange(trainingSize, filteredTargets.rows);
        // testSamples.push_back(samples.rowRange(8000, 13000));
        // testTarget.push_back(target.rowRange(8000, 13000));
    }

    // use = 1 if you want to apply the calculated Standartizations parameters on another set
    void standardizeData(cv::Mat &data, bool use) // standardize
    {
        // Initialize mean and standard deviation matrices only if 'use' is false
        if (!use)
        {
            // it should theoreticaly be possible to calc mean and stdDev of a sample and use these to standardize the rest of the set
            // currently not working
            mean = cv::Mat::zeros(1, data.cols, CV_32F);
            stdDev = cv::Mat::zeros(1, data.cols, CV_32F);
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
    void saveAsCsv(cv::Mat targets_, cv::Mat samples_, std::string Name)
    {
        cv::Mat data_;
        cv::hconcat(targets_, samples_, data_);
        std::ofstream outputFile(Name + ".csv");
        outputFile << format(data_, cv::Formatter::FMT_CSV) << std::endl;
        outputFile.close();
    }

    void testPrint(cv::Mat &data, bool sizeOnly)
    {
        if (!sizeOnly)
        {
            for (int i = 0; i < data.rows; ++i)
            {
                std::cout << data.at<float>(i, 0) << " ";
            }
        }
        std::cout << data.size() << std::endl;
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

    cv::Mat standatisationSet; // extra set to calculate mean and stdDev
    cv::Mat trainSamples;
    cv::Mat testSamples;
    cv::Mat testTarget;
    cv::Mat trainTarget;

    cv::Scalar meanScalar, stdDevScalar;
};

class PCA_
{
public:
    PCA_() {}
    ~PCA_() {}
    void applyPCA(cv::Mat &data, int components = 1000)
    {
        cv::PCA pca_analysis(data, cv::Mat(), cv::PCA::DATA_AS_ROW, components);
        data = pca_analysis.project(data);
    }

private:
};

class SVM
{
public:
    SVM() {}
    ~SVM() {}

    void trainSVM(cv::Mat &trainLabels, cv::Mat &trainData, double C, double gamma)
    {
        trainLabels.convertTo(trainLabels, CV_32S);
        svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(C);
        svm->setGamma(gamma);
        std::cout << "training \n";
        svm->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);
    }

    float evaluateSVM(cv::Mat &testData, cv::Mat &testLabels)
    {
        if (testData.empty() || testLabels.empty())
        {
            std::cerr << "Error: TestData or TestLabels are empty." << std::endl;
            return 0.0; // or handle more appropriately depending on your application
        }

        if (!svm || !svm->isTrained())
        {
            std::cerr << "Error: SVM model is not trained or not initialized." << std::endl;
            return 0.0;
        }

        cv::Mat response;
        try
        {
            svm->predict(testData, response);
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "OpenCV Error during SVM prediction: " << e.what() << std::endl;
            return 0.0;
        }

        int correctPredictions = cv::countNonZero(response == testLabels);
        return correctPredictions / (float)testData.rows;
    }

    void gridSearch(cv::Mat &trainLabels, cv::Mat &trainData, cv::Mat &testLabels, cv::Mat &testData)
    {
        std::vector<double> C_values = {0.4, 0.5, 0.6, 0.7, 1, 2, 3};
        std::vector<double> gamma_values = {0.02, 0.01, 0.001, 0.005, 0.0001};

        double bestC = 0, bestGamma = 0;
        float bestAccuracy = 0;

        for (double C : C_values)
        {
            for (double gamma : gamma_values)
            {
                trainSVM(trainLabels, trainData, C, gamma);
                float accuracy = evaluateSVM(testData, testLabels);
                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    bestC = C;
                    bestGamma = gamma;
                }
            }
        }

        std::cout << "Best C: " << bestC << ", Best Gamma: " << bestGamma << ", Best Accuracy: " << bestAccuracy << std::endl;

        // Optionally, you can retrain the SVM with the best parameters on the full dataset here
    }

private:
    cv::Ptr<cv::ml::SVM> svm;
};

int main()
{
    dataProcessing dp;
    dp.loadData();
    dp.mergeAndShuffleData(); // shuffle whole Dataset
    dp.sampleAndFilter();

    cv::Mat trainingTargets = dp.getTarget(1);
    dp.testPrint(trainingTargets, 1);
    cv::Mat trainingSamples = dp.getSamples(1);
    dp.testPrint(trainingSamples, 1);

    cv::Mat testingTargets = dp.getTarget(2);
    cv::Mat testingSamples = dp.getSamples(2);

    dp.standardizeData(trainingSamples, 0);
    std::cout << "Training Dataset metrics:";
    dp.testPrint(trainingSamples, 1);
    dp.testPrintMeanStdDev(trainingSamples, 1);

    dp.saveAsCsv(trainingTargets, trainingSamples, "Standardized_training");

    dp.standardizeData(testingSamples, 0);
    std::cout << "Testing Dataset metrics:";
    dp.testPrint(testingSamples, 1);
    dp.testPrintMeanStdDev(testingSamples, 1);

    PCA_ pca;
    pca.applyPCA(trainingSamples);
    std::cout << "Training Dataset metrics after PCA:";
    // dp.standardizeData(trainingSamples, 0);
    dp.testPrint(trainingSamples, 1);
    dp.testPrintMeanStdDev(trainingSamples, 1);

    dp.saveAsCsv(trainingTargets, trainingSamples, "PCA");

    SVM svmModel;

    svmModel.trainSVM(trainingTargets, trainingSamples, 1.0, 0.01);
    float accuracy = svmModel.evaluateSVM(testingSamples, testingTargets);
    std::cout << "Initial SVM Accuracy: " << accuracy << std::endl;

    svmModel.gridSearch(trainingTargets, trainingSamples, testingTargets, testingSamples);

    accuracy = svmModel.evaluateSVM(testingSamples, testingTargets);
    std::cout << "Optimized SVM Accuracy: " << accuracy << std::endl;
    return 0;
};