﻿// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "trackerPrecomp.hpp"

#include "trackerXFDScaleEstimation.hpp"
#include "trackerXFDUtils.hpp"

//Discriminative Scale Space Tracking
namespace cv
{

class ParallelGetScaleFeatures : public ParallelLoopBody
{
public:
    ParallelGetScaleFeatures(
        Mat img,
        Point2f pos,
        Size2f base_target_sz,
        float current_scale,
        std::vector<float> &scale_factors,
        Mat scale_window,
        Size scale_model_sz,
        int col_len,
        Mat &result)
    {
        this->img = img;
        this->pos = pos;
        this->base_target_sz = base_target_sz;
        this->current_scale = current_scale;
        this->scale_factors = scale_factors;
        this->scale_window = scale_window;
        this->scale_model_sz = scale_model_sz;
        this->col_len = col_len;
        this->result = result;
    }
    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        for (int s = range.start; s < range.end; s++) 
		{
            Size patch_sz = Size(static_cast<int>(current_scale * scale_factors[s] * base_target_sz.width),
                    static_cast<int>(current_scale * scale_factors[s] * base_target_sz.height));
            Mat img_patch = get_subwindow(img, pos, patch_sz.width, patch_sz.height);
            img_patch.convertTo(img_patch, CV_32FC3);
            resize(img_patch, img_patch, Size(scale_model_sz.width, scale_model_sz.height),0,0,INTER_LINEAR);

			std::vector<Mat> hog;
			hog = get_features_hog(img_patch, 1);

			for (int i = 0; i < static_cast<int>(hog.size()); ++i)
			{
				hog[i] = hog[i].t();
				hog[i] = scale_window.at<float>(0, s) * hog[i].reshape(0, col_len);
				hog[i].copyTo(result(Rect(Point(s, i*col_len), hog[i].size())));
			}

			Mat gray_m;
			cvtColor(img_patch, gray_m, COLOR_BGR2GRAY);
			gray_m.convertTo(gray_m, CV_32FC1, 1.0 / 255.0, -0.5);

			gray_m = gray_m.t();
			gray_m = scale_window.at<float>(0, s) * gray_m.reshape(0, col_len);
			gray_m.copyTo(result(Rect(Point(s, hog.size()*col_len), gray_m.size())));
        }
    }

    ParallelGetScaleFeatures& operator=(const ParallelGetScaleFeatures &) {
        return *this;
    }

private:
    Mat img;
    Point2f pos;
    Size2f base_target_sz;
    float current_scale;
    std::vector<float> scale_factors;
    Mat scale_window;
    Size scale_model_sz;
    int col_len;
    Mat result;
};


DSST::DSST(const Mat &image,
        Rect2f bounding_box,
        Size2f template_size,
        int numberOfScales,
        float scaleStep,
        float maxModelArea,
        float sigmaFactor,
        float scaleLearnRate):
    scales_count(numberOfScales), scale_step(scaleStep), max_model_area(maxModelArea),
    sigma_factor(sigmaFactor), learn_rate(scaleLearnRate)
{
    original_targ_sz = bounding_box.size();
    Point2f object_center = Point2f(bounding_box.x + original_targ_sz.width / 2,
            bounding_box.y + original_targ_sz.height / 2);

    current_scale_factor = 1.0;
    if(scales_count % 2 == 0)
        scales_count++;

    scale_sigma = static_cast<float>(sqrt(scales_count) * sigma_factor);

    min_scale_factor = pow(scale_step,
            cvCeil(log(max(5.0 / template_size.width, 5.0 / template_size.height)) / log(scale_step)));
    max_scale_factor = powf(scale_step,
            static_cast<float>(cvFloor(log(min((float)image.rows / (float)bounding_box.width,
            (float)image.cols / (float)bounding_box.height)) / log(scale_step))));
    ys = Mat(1, scales_count, CV_32FC1);
    float ss, sf;
    for(int i = 0; i < ys.cols; ++i) 
	{
        ss = (float)(i+1) - cvCeil((float)scales_count / 2.0f);
        ys.at<float>(0,i) = static_cast<float>(exp(-0.5 * pow(ss,2) / pow(scale_sigma,2)));
        sf = static_cast<float>(i + 1);
        scale_factors.push_back(pow(scale_step, cvCeil((   float)scales_count / 2.0f) - sf));
    }

    scale_window = get_hann_win(Size(scales_count, 1));

    float scale_model_factor = 1.0;
    if(template_size.width * template_size.height * pow(scale_model_factor, 2) > max_model_area)
    {
        scale_model_factor = sqrt(max_model_area /
                (template_size.width * template_size.height));
    }
    scale_model_sz = Size(cvFloor(template_size.width * scale_model_factor),
            cvFloor(template_size.height * scale_model_factor));

    Mat scale_resp = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);

    Mat ysf_row = Mat(ys.size(), CV_32FC2);
    dft(ys, ysf_row, DFT_ROWS | DFT_COMPLEX_OUTPUT, 0);
    ysf = repeat(ysf_row, scale_resp.rows, 1);
    Mat Fscale_resp;
    dft(scale_resp, Fscale_resp, DFT_ROWS | DFT_COMPLEX_OUTPUT);
    mulSpectrums(ysf, Fscale_resp, sf_num, 0 , true);
    Mat sf_den_all;
    mulSpectrums(Fscale_resp, Fscale_resp, sf_den_all, 0, true);
    reduce(sf_den_all, sf_den, 0, REDUCE_SUM, -1);
}

DSST::~DSST()
{
}

Mat DSST::get_scale_features(
        Mat img,
        Point2f pos,
        Size2f base_target_sz,
        float current_scale,
        std::vector<float> &scale_factors,
        Mat scale_window,
        Size scale_model_sz)
{
    Mat result;
    int col_len = 0;
    Size patch_sz = Size(cvFloor(current_scale * scale_factors[0] * base_target_sz.width),
            cvFloor(current_scale * scale_factors[0] * base_target_sz.height));
    Mat img_patch = get_subwindow(img, pos, patch_sz.width, patch_sz.height);
    img_patch.convertTo(img_patch, CV_32FC3);
    resize(img_patch, img_patch, Size(scale_model_sz.width, scale_model_sz.height),0,0,INTER_LINEAR);
	
	std::vector<Mat> hog;
	hog = get_features_hog(img_patch, 1);
	result = Mat(Size((int)scale_factors.size(), hog[0].cols * hog[0].rows * (int)(hog.size() + 1)), CV_32F);
	std::cout << "dsst_feature:  Size(scale_model_sz.width, scale_model_sz.height) = " << Size(scale_model_sz.width, scale_model_sz.height) << " img_patch =  " << img_patch.size() << "hog" << ysf.size() << std::endl;

	col_len = hog[0].cols * hog[0].rows;
	for (int i = 0; i < static_cast<int>(hog.size()); ++i) {
		hog[i] = hog[i].t();
		hog[i] = scale_window.at<float>(0, 0) * hog[i].reshape(0, col_len);
		hog[i].copyTo(result(Rect(Point(0, i*col_len), hog[i].size())));
	}

	Mat gray_m;
	cvtColor(img_patch, gray_m, COLOR_BGR2GRAY);
	gray_m.convertTo(gray_m, CV_32FC1, 1.0 / 255.0, -0.5);
	//result = Mat(Size((int)scale_factors.size(), gray_m.cols * gray_m.rows), CV_32F);
	col_len = gray_m.cols * gray_m.rows;
	gray_m = gray_m.t();
	gray_m = scale_window.at<float>(0, 0) * gray_m.reshape(0, col_len);
	gray_m.copyTo(result(Rect(Point(0, hog.size()*col_len), gray_m.size())));

    ParallelGetScaleFeatures parallelGetScaleFeatures(img, pos, base_target_sz,
            current_scale, scale_factors, scale_window, scale_model_sz, col_len, result);
    parallel_for_(Range(1, static_cast<int>(scale_factors.size())), parallelGetScaleFeatures);
    return result;
}

void DSST::update(const Mat &image, const Point2f object_center)
{
    Mat scale_features = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);
    Mat Fscale_features;
    dft(scale_features, Fscale_features, DFT_ROWS | DFT_COMPLEX_OUTPUT);
    Mat new_sf_num;
    Mat new_sf_den;
    Mat new_sf_den_all;
    mulSpectrums(ysf, Fscale_features, new_sf_num, DFT_ROWS, true);
    Mat sf_den_all;
    mulSpectrums(Fscale_features, Fscale_features, new_sf_den_all, DFT_ROWS, true);
    reduce(new_sf_den_all, new_sf_den, 0, REDUCE_SUM, -1);
	std::cout << "dsst_update:  scale_features = " << scale_features.size() << " Fscale_features =  " << Fscale_features.size() << "ysf" << ysf.size() << std::endl;
    sf_num = (1 - learn_rate) * sf_num + learn_rate * new_sf_num;
    sf_den = (1 - learn_rate) * sf_den + learn_rate * new_sf_den;
}

float DSST::getScale(const Mat &image, const Point2f object_center)
{
    Mat scale_features = get_scale_features(image, object_center, original_targ_sz,
            current_scale_factor, scale_factors, scale_window, scale_model_sz);

    Mat Fscale_features;
    dft(scale_features, Fscale_features, DFT_ROWS | DFT_COMPLEX_OUTPUT);

    mulSpectrums(Fscale_features, sf_num, Fscale_features, 0, false);
	
    Mat scale_resp;
    reduce(Fscale_features, scale_resp, 0, REDUCE_SUM, -1);
    scale_resp = divide_complex_matrices(scale_resp, sf_den + 0.01f);
    idft(scale_resp, scale_resp, DFT_REAL_OUTPUT|DFT_SCALE);
    Point max_loc;
    minMaxLoc(scale_resp, NULL, NULL, NULL, &max_loc);
	std::cout << "dsst_getscale:  scale_features = " << scale_features.size() << " Fscale_features =  " << Fscale_features.size() << "scale_resp" << scale_resp.size() << std::endl;
    current_scale_factor *= scale_factors[max_loc.x];
    if(current_scale_factor < min_scale_factor)
        current_scale_factor = min_scale_factor;
    else if(current_scale_factor > max_scale_factor)
        current_scale_factor = max_scale_factor;

    return current_scale_factor;
}
} /* namespace cv */
