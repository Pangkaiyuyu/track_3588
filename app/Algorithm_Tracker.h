// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef ALGORITHM_TRACKER_H
#define ALGORITHM_TRACKER_H

#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"

namespace cv 
{
	inline namespace tracking 
	{

		class CV_EXPORTS_W TrackerXFD : public Tracker
		{
			protected:
				TrackerXFD();
			public:
				virtual ~TrackerXFD() CV_OVERRIDE;
				struct CV_EXPORTS_W_SIMPLE Params
				{
					CV_WRAP Params();

					CV_PROP_RW bool use_hog;
					CV_PROP_RW bool use_cnn;
					CV_PROP_RW bool use_color_names;
					CV_PROP_RW bool use_gray;
					CV_PROP_RW bool use_rgb;
					CV_PROP_RW bool use_channel_weights;
					CV_PROP_RW bool use_segmentation;

					CV_PROP_RW std::string window_function; //!<  Window function: "hann", "cheb", "kaiser"
					CV_PROP_RW float kaiser_alpha;
					CV_PROP_RW float cheb_attenuation;

					CV_PROP_RW float template_size;
					CV_PROP_RW float gsl_sigma;
					CV_PROP_RW float hog_orientations;
					CV_PROP_RW float hog_clip;
					CV_PROP_RW float padding;
					CV_PROP_RW float filter_lr;
					CV_PROP_RW float weights_lr;
					CV_PROP_RW int num_hog_channels_used;
					CV_PROP_RW int num_cnn_channels_used;
					CV_PROP_RW int admm_iterations;
					CV_PROP_RW int histogram_bins;
					CV_PROP_RW float histogram_lr;
					CV_PROP_RW int background_ratio;
					CV_PROP_RW int number_of_scales;
					CV_PROP_RW float scale_sigma_factor;
					CV_PROP_RW float scale_model_max_area;
					CV_PROP_RW float scale_lr;
					CV_PROP_RW float scale_step;

					CV_PROP_RW float psr_threshold; //!< we lost the target, if the psr is lower than this.
					CV_PROP_RW float score_threshold;
				};
				static CV_WRAP Ptr<TrackerXFD> create(const TrackerXFD::Params &parameters = TrackerXFD::Params());
				CV_WRAP virtual void setInitialMask(InputArray mask) = 0;

				CV_WRAP virtual void setOutThresholdParm(float th_psr, float th_scr) = 0;

				CV_WRAP virtual bool getTrackerStatue() = 0;
		};

		class CV_EXPORTS_W TrackerKCF : public Tracker
		{
			protected:
				TrackerKCF();
			public:
				virtual ~TrackerKCF() CV_OVERRIDE;

				enum MODE {
				  GRAY   = (1 << 0),
				  CN     = (1 << 1),
				  CUSTOM = (1 << 2)
				};

				struct CV_EXPORTS_W_SIMPLE Params
				{
					CV_WRAP Params();

					CV_PROP_RW float detect_thresh;         //!<  detection confidence threshold
					CV_PROP_RW float sigma;                 //!<  gaussian kernel bandwidth
					CV_PROP_RW float lambda;                //!<  regularization
					CV_PROP_RW float interp_factor;         //!<  linear interpolation factor for adaptation
					CV_PROP_RW float output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
					CV_PROP_RW float pca_learning_rate;     //!<  compression learning rate
					CV_PROP_RW bool resize;                  //!<  activate the resize feature to improve the processing speed
					CV_PROP_RW bool split_coeff;             //!<  split the training coefficients into two matrices
					CV_PROP_RW bool wrap_kernel;             //!<  wrap around the kernel values
					CV_PROP_RW bool compress_feature;        //!<  activate the pca method to compress the features
					CV_PROP_RW int max_patch_size;           //!<  threshold for the ROI size
					CV_PROP_RW int compressed_size;          //!<  feature size after compression
					CV_PROP_RW int desc_pca;        //!<  compressed descriptors of TrackerKCF::MODE
					CV_PROP_RW int desc_npca;       //!<  non-compressed descriptors of TrackerKCF::MODE
				};

				static CV_WRAP
				Ptr<TrackerKCF> create(const TrackerKCF::Params &parameters = TrackerKCF::Params());
				typedef void (*FeatureExtractorCallbackFN)(const Mat, const Rect, Mat&);
				virtual void setFeatureExtractor(FeatureExtractorCallbackFN callback, bool pca_func = false) = 0;
		};

	}
}

#endif // ALGORITHM_TRACKER_H
