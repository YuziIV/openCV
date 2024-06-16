#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

class imageHandler
{
public:
    imageHandler(){};
    ~imageHandler()
    {
        std::cout << "ImageHandler Destructed\n";
    };

    cv::Mat readImg(std::string Path, bool resize)
    {
        name_ = "Source Image";
        // Read the image from the specified path
        srcImg = cv::imread(Path, cv::IMREAD_COLOR);

        // Check if the image was successfully loaded
        if (srcImg.empty())
        {
            std::cout << "Could not read the image: " << Path << std::endl;
            return cv::Mat();
        }
        else if (resize == 1)
        {

            cv::resize(srcImg, resizedImg, cv::Size(), 0.3, 0.3);
            segmentedImg = cv::Mat::zeros(resizedImg.size(), CV_8U);
            return resizedImg;
        }
        else
        {
            return srcImg;
        }
    }

    // void segmentImg(cv::Mat srcImg)
    // {

    //     cv::cvtColor(srcImg, HSVImg, cv::COLOR_BGR2HSV); // convert Colorspace from RGB to HSV
    //     // Thresholding the HSV image based on trackbar positions
    //     // Setting the lower boundries with the slider_Vals
    //     // Setting the upper boundries with the max_Vals
    //     // For dynamic upper/lower Boundary changes, change the max_Values to TB_maxVal and their corr. names
    //     // cv::inRange(HSVImg, cv::Scalar(TBHueVal, TBSatVal, TBValueVal), cv::Scalar(maxHue, maxSaturation, maxValue), segmentedImg);
    //     cv::inRange(HSVImg, cv::Scalar(TBHueVal, TBSatVal, TBValueVal), cv::Scalar(maxHue, maxSaturation, maxValue), tempMask);
    //     srcImg.copyTo(segmentedImg ,tempMask);
    // }

    // void updateSegmentationDisplay()
    // {
    // segmentImg(srcImg); // Use the current class's srcImg
    // cv::imshow("HSV Raum", segmentedImg);
    // }

    static void updateCallback(int, void *Userdata)
    {
        imageHandler *instance = (imageHandler *)Userdata;
        cv::cvtColor(instance->resizedImg, instance->HSVImg, cv::COLOR_BGR2HSV);
        // Thresholding the HSV image based on trackbar positions
        // Setting the lower boundries with the slider_Vals
        // Setting the upper boundries with the max_Vals
        // For dynamic upper/lower Boundary changes, change the max_Values to TB_maxVal and their corr. names
        // cv::inRange(HSVImg, cv::Scalar(TBHueVal, TBSatVal, TBValueVal), cv::Scalar(maxHue, maxSaturation, maxValue), segmentedImg);
        cv::inRange(instance->HSVImg, cv::Scalar(instance->TBHueVal, instance->TBSatVal, instance->TBValueVal), cv::Scalar(instance->TBHueValmax, instance->maxSaturation, instance->maxValue), instance->tempMask);
        instance->segmentedImg = cv::Mat::zeros(instance->resizedImg.size(), CV_32F);
        instance->resizedImg.copyTo(instance->segmentedImg, instance->tempMask);

        cv::imshow("HSV Raum", instance->segmentedImg);
        cv::cvtColor(instance->resizedImg, instance->HSVImg, cv::COLOR_BGR2HSV); // convert Colorspace from RGB to HSV
    }

    void createTrackbars()
    {
        name_ = "HSV Raum";
        cv::namedWindow(name_);

        // Creating trackbars for each HSV component
        // change the names ot the lower boundaries by appending "_lower" and uncomment the following 3 lines
        cv::createTrackbar("Hue_upper", name_, &TBHueValmax, maxHue, updateCallback, this);
        cv::createTrackbar("Hue_lower", name_, &TBHueVal, maxHue, updateCallback, this);

        cv::createTrackbar("Saturation_upper", name_, &TBSatValmax, maxSaturation, updateCallback, this);
        cv::createTrackbar("Saturation_lower", name_, &TBSatVal, maxSaturation, updateCallback, this);

        cv::createTrackbar("Value_upper", name_, &TBValueValmax, maxValue, updateCallback, this);
        cv::createTrackbar("Value_lower", name_, &TBValueVal, maxValue, updateCallback, this);

        // adapt this function as well
        cv::setTrackbarPos("Hue_upper", name_, 60);
        cv::setTrackbarPos("Hue_lower", name_, 50);

        cv::setTrackbarPos("Saturation_upper", name_, 255);
        cv::setTrackbarPos("Saturation_lower", name_, 125);

        cv::setTrackbarPos("Value_upper", name_, 255);
        cv::setTrackbarPos("Value_lower", name_, 144);
    }

    cv::Mat mergeSobel8U(const cv::Mat &sobelX, const cv::Mat &sobelY)
    {
        CV_Assert(sobelX.size() == sobelY.size() && sobelX.type() == sobelY.type() && sobelX.type() == CV_8U);

        // Create a new image to store the combined result, same type as the inputs
        cv::Mat merged = cv::Mat::zeros(sobelX.size(), sobelX.type());

        // Calculate the gradient magnitude for each pixel
        for (int i = 0; i < sobelX.rows; ++i)
        {
            for (int j = 0; j < sobelX.cols; ++j)
            {
                // No need to convert types since we're already working with CV_8U
                uchar gx = sobelX.at<uchar>(i, j);
                uchar gy = sobelY.at<uchar>(i, j);
                // Use min to ensure we don't exceed the 255 limit for uchar
                merged.at<uchar>(i, j) = std::min(static_cast<int>(std::sqrt(gx * gx + gy * gy)), 255);
            }
        }

        return merged;
    }

    cv::Mat detectCircles(cv::Mat binaryImg, cv::Mat MatToProjectOn)
    {
        // cv::imshow("To perform from",binaryImg);
        // cv::imshow("To perform on",MatToProjectOn);
        std::vector<cv::Vec3f> circlesArray;
        cv::HoughCircles(binaryImg, circlesArray, cv::HOUGH_GRADIENT, 3, 500, 50, 20, 20, 50);

        // if (circlesArray.empty())
        // {
        //     std::cout << "ARRAY EMPTY \n";
        // }
        // else
        // {
        //     for (const auto &temp : circlesArray)
        //     {
        //         std::cout << temp << "\n";
        //     }
        //     std::cout << "array size:" << circlesArray.size();
        // }

        // Drawing the detected circles on the original (or processed) image
        cv::Mat circleImg = MatToProjectOn.clone();
        for (size_t i = 0; i < circlesArray.size(); i++)
        {
            cv::Point center(cvRound(circlesArray[i][0]), cvRound(circlesArray[i][1]));
            int radius = cvRound(circlesArray[i][2]);
            // Circle center
            cv::circle(circleImg, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
            // Circle outline
            cv::circle(circleImg, center, radius, cv::Scalar(255, 0, 0), 3, 8, 0);
        }
        return circleImg;
    }
    cv::Mat detectRectangels(cv::Mat binaryImg, cv::Mat MatToProjectOn)
    {
        // cv::imshow("To perform from",binaryImg);
        // cv::imshow("To perform on",MatToProjectOn);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(binaryImg, lines, 1, CV_PI / 180, 50, 50, 10);

        // if (circlesArray.empty())
        // {
        //     std::cout << "ARRAY EMPTY \n";
        // }
        // else
        // {
        //     for (const auto &temp : circlesArray)
        //     {
        //         std::cout << temp << "\n";
        //     }
        //     std::cout << "array size:" << circlesArray.size();
        // }

        // Drawing the detected circles on the original (or processed) image
        cv::Mat rectImg = MatToProjectOn.clone();
        for (size_t i = 0; i < lines.size(); i++)
        {
            for (size_t j = i + 1; j < lines.size(); j++)
            {
                cv::Point2f pt1(lines[i][0], lines[i][1]);
                cv::Point2f pt2(lines[i][2], lines[i][3]);
                cv::Point2f pt3(lines[j][0], lines[j][1]);
                cv::Point2f pt4(lines[j][2], lines[j][3]);

                // Check if the lines are perpendicular
                if (fabs(cv::fastAtan2(pt2.y - pt1.y, pt2.x - pt1.x) - cv::fastAtan2(pt4.y - pt3.y, pt4.x - pt3.x)) > 80 &&
                    fabs(cv::fastAtan2(pt2.y - pt1.y, pt2.x - pt1.x) - cv::fastAtan2(pt4.y - pt3.y, pt4.x - pt3.x)) < 100)
                {
                    cv::line(rectImg, pt1, pt2, cv::Scalar(0, 0, 255), 2);
                    cv::line(rectImg, pt3, pt4, cv::Scalar(0, 0, 255), 2);
                }
            }
        }
        return rectImg;
    }
    cv::Mat getSegmentedImg() const
    {
        return this->segmentedImg;
    }

private:
    cv::Mat tempMask;
    cv::Mat srcImg;
    cv::Mat resizedImg;
    cv::Mat HSVImg;
    cv::Mat segmentedImg = cv::Mat::zeros(resizedImg.size(), CV_32F);

    int TBHueVal;
    int TBSatVal;
    int TBValueVal;

    // Values for inital limit set only needed for the right numbers to be displayed
    int TBHueValmax = 180;
    int TBSatValmax = 255;
    int TBValueValmax = 255;

    int maxHue = 180;
    int maxSaturation = 255;
    int maxValue = 255;

    std::string name_;
};
/*______________________________________________________________________________________________________
________________________________________________________________________________________________________*/
class discreteConvolution
{
public:
    discreteConvolution(cv::Mat k, cv::Mat &segmentedImg) : Kernel(k), inputImg(segmentedImg)
    {
        conv(Kernel, inputImg);
    };
    void showKernel()
    {
        cv::imshow("Kernel", Kernel);
    }

    void showOutput(std::string outputName)
    {
        cv::imshow(outputName, output8U);
    }
    cv::Mat getOutput()
    {
        return output8U;
    }

    void conv(cv::Mat kernelMat, cv::Mat inputMat)
    {
        /////////// Preprocesseing Input Matrix/////////////
        // Greying the Image
        cv::Mat greyMat;
        if (inputMat.channels() > 1)
        {
            cv::cvtColor(inputMat, greyMat, CV_BGR2GRAY);
        }
        else
        {
            greyMat = inputMat;
        }
        cv::Mat inputFloat;

        greyMat.convertTo(inputFloat, CV_32F);
        // Add a border to the input image to handle edge pixels during convolution
        cv::Mat matWithBorder;
        int borderSize = kernelMat.rows / 2; // Assuming the kernel is square and has an odd dimension
        cv::copyMakeBorder(inputFloat, matWithBorder, borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);

        // Create an output image of the same size as the input, but with float precision
        cv::Mat outputImg = cv::Mat::zeros(inputMat.size(), CV_32F);

        // // Perform convolution
        for (int i = borderSize; i < matWithBorder.rows - borderSize; ++i)
        {
            for (int j = borderSize; j < matWithBorder.cols - borderSize; ++j)
            {
                float sum = 0.0f;
                for (int k = -borderSize; k <= borderSize; ++k)
                {
                    for (int l = -borderSize; l <= borderSize; ++l)
                    {
                        float imageValue = matWithBorder.at<float>(i + k, j + l);
                        // Adjust kernel indexing to account for kernel size and ensure correct center alignment
                        float kernelValue = kernelMat.at<float>(k + borderSize, l + borderSize);
                        sum += imageValue * kernelValue;
                    }
                }
                // Assign the computed sum to the output image, adjusting indices to account for added border
                if (sum < 0)
                {
                    sum = abs(sum);
                }
                outputImg.at<float>(i - borderSize, j - borderSize) = sum;
            }
        }

        // Normalize to 0-255 and convert to CV_8U
        cv::Mat normalizedOutput;
        cv::normalize(outputImg, normalizedOutput, 0, 255, cv::NORM_MINMAX);
        normalizedOutput.convertTo(output8U, CV_8U);

        // return output8U;
    }

private:
    cv::Mat Kernel;
    cv::Mat &inputImg;
    cv::Mat output8U;
};

/*______________________________________________________________________________________________________
________________________________________________________________________________________________________*/

class createKernel
{
public:
    cv::Mat meanKernel(int dim_X, int dim_Y, double scalar)
    {
        cv::Mat Ones = cv::Mat::ones(dim_X, dim_Y, CV_32F);
        cv::Mat meanKernel = scalar * Ones;
        return meanKernel;
    }

    cv::Mat gaussKernel(int dim, double sigma)
    {
        // Generate a 1D Gaussian kernel
        cv::Mat gaussKernelX = cv::getGaussianKernel(dim, sigma, CV_32F);
        // Use the outer product to create a 2D Gaussian kernel from the 1D kernels
        cv::Mat gaussKernel2D = gaussKernelX * gaussKernelX.t(); // t() is transpose
        return gaussKernel2D;
    }

    cv::Mat sobelKernelX()
    {
        // Sobel kernel for horizontal edge detection
        cv::Mat Gx = (cv::Mat_<float>(3, 3) << 1, 0, -1,
                      2, 0, -2,
                      1, 0, -1);
        return Gx;
    }

    cv::Mat sobelKernelY()
    {
        // Sobel kernel for vertical edge detection
        cv::Mat Gy = (cv::Mat_<float>(3, 3) << 1, 2, 1,
                      0, 0, 0,
                      -1, -2, -1);
        return Gy;
    }

private:
};

int main()
{
    imageHandler img;
    cv::Mat src = img.readImg("./rect.jpg", 1);
    // cv::imshow("Quelle", src);
    img.createTrackbars();
    cv::Mat SegImg = img.getSegmentedImg();

    createKernel K;
    // cv::Mat Kernel = K.meanKernel(3, 3, 1.0 / 9.0);
    cv::Mat Kernel = K.gaussKernel(3, 1 / 9);
    discreteConvolution Blur(Kernel, SegImg);
    Blur.showOutput("Gaussian Blurred");
    cv::Mat blurOutput = Blur.getOutput();

    cv::Mat horEdgeKernel = K.sobelKernelX();
    discreteConvolution sobelXEdge(horEdgeKernel, blurOutput);
    sobelXEdge.showOutput("Horizontal Detection");
    cv::Mat sobelX = sobelXEdge.getOutput();

    cv::Mat verEdgeKernel = K.sobelKernelY();
    discreteConvolution sobelYEdge(verEdgeKernel, blurOutput);
    sobelYEdge.showOutput("Vertical Detection");
    cv::Mat sobelY = sobelYEdge.getOutput();

    cv::Mat mergedSobel8U = img.mergeSobel8U(sobelX, sobelY);
    cv::Mat binaryImg;
    cv::imshow("Merged Sobel Edge detection", mergedSobel8U);
    cv::threshold(mergedSobel8U, binaryImg, 0, 255, cv::THRESH_OTSU);
    cv::imshow("Otsu Thresholding", binaryImg);

    // cv::Mat HughCircle = img.detectCircles(binaryImg, src);
    // cv::imshow("Hugh Circle detection", HughCircle);

    cv::Mat HughRect = img.detectRectangels(binaryImg, src);
    cv::imshow("Hugh rectangle detection", HughRect);

    cv::waitKey(0);
    return 0;
}