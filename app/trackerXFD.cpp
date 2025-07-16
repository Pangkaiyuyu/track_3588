//#define WIN_X64

#ifdef WIN_X64
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#endif

#include "trackerPrecomp.hpp"
#include "trackerXFDSegmentation.hpp"
#include "trackerXFDUtils.hpp"
#include "trackerXFDScaleEstimation.hpp"

namespace cv {
inline namespace tracking {
namespace impl {

/**
* \brief Implementation of TrackerModel for XFD algorithm
*/
class TrackerXFDModel CV_FINAL : public TrackerModel
{
public:
    TrackerXFDModel(){}
    ~TrackerXFDModel(){}
protected:
    void modelEstimationImpl(const std::vector<Mat>& /*responses*/) CV_OVERRIDE {}
    void modelUpdateImpl() CV_OVERRIDE {}
};

class TrackerXFDImpl CV_FINAL : public TrackerXFD
{
public:
    TrackerXFDImpl(const Params &parameters = Params());

    Params params;

    Ptr<TrackerXFDModel> model;

    // Tracker API
    virtual void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    virtual bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
    virtual void setInitialMask(InputArray mask) CV_OVERRIDE;

	//Extar API（给用户提供外部设置参数接口）
	void setOutThresholdParm(float th_psr, float th_scr) CV_OVERRIDE;
	bool getTrackerStatue() CV_OVERRIDE;

protected:
    void update_xfd_filter(const Mat &image, const Mat &my_mask);
    void update_histograms(const Mat &image, const Rect &region);
    void extract_histograms(const Mat &image, cv::Rect region, Histogram &hf, Histogram &hb);
    std::vector<Mat> create_xfd_filter(const std::vector<cv::Mat>
            img_features, const cv::Mat Y, const cv::Mat P);
    Mat calculate_response(const Mat &image, const std::vector<Mat> filter);
    Mat get_location_prior(const Rect roi, const Size2f target_size, const Size img_sz);
    Mat segment_region(const Mat &image, const Point2f &object_center,
            const Size2f &template_size, const Size &target_size, float scale_factor);

	int imregionalmax(Mat input, int num_LocMax, float threshold, 
		int minDistBtwLocMax, int assignSecondNum, double &assignSecondMaximum);

    Point2f estimate_new_position(const Mat &image);
    std::vector<Mat> get_features(const Mat &patch, const Size2i &feature_size);

    bool check_mask_area(const Mat &mat, const double obj_area);
	double calculate_resp_quality(const Mat &image, Point max_loc, double max_val);
	void calculate_resp_score(double resp_quality);
	bool judge_loss_Target(double response_max, double response_score, double secondmax_response);

#ifdef WIN_X64
	static DWORD WINAPI DealProc(LPVOID lpParam);
	static DWORD WINAPI DealProc2(LPVOID lpParam);
	CRITICAL_SECTION g_cs;

#else
	static void * DealProc(void * lpParam);
	static void * DealProc2(void * lpParam);
	pthread_mutex_t m_mutex_tracker;
#endif

	
    float current_scale_factor;
    Mat window;
    Mat yf;
    Rect2f bounding_box;
    std::vector<Mat> xfd_filter;
    std::vector<float> filter_weights;
    Size2f original_target_size;   
    Size2i image_size;
    Size2f template_size;
    Size2i rescaled_template_size;
    float rescale_ratio;
    Point2f object_center;
    DSST dsst;
    Histogram hist_foreground;
    Histogram hist_background;
    double p_b;
    Mat erode_element;
    Mat filter_mask;
    Mat preset_mask;
    Mat default_mask;
    float default_mask_area;
    int cell_size;

	//并行操作以及判丢附加变量//
	bool flag_loss_target;

	int update_flag;
	int interrupt_flag;

	int dsst_flag;
	int interrupt2_flag;

	Mat img_now;
	int Resp_num;
	double resp_quality_init;
	double resp_norm;
	double resp_mean;
	double Resp_score;

};

// TrackerXFDImpl::TrackerXFDImpl(const TrackerXFD::Params &parameters) : params(parameters)
// {
	
// 	update_flag = 0;
// 	interrupt_flag = 0;

// 	dsst_flag = 0;
// 	interrupt2_flag = 0;

// #ifdef WIN_X64

// 	InitializeCriticalSection(&g_cs);

// 	HANDLE hThead1 = CreateThread(NULL, 0, DealProc, this, 0, NULL);
// 	HANDLE hThead2 = CreateThread(NULL, 0, DealProc2, this, 0, NULL);

// #else

// 	pthread_mutex_init(&m_mutex_tracker, NULL);

// 	int m_inher_trains = 0;
// 	pthread_attr_t m_attr_trains;
// 	struct sched_param m_param_trains;

// 	memset(&m_attr_trains, 0, sizeof(pthread_attr_t));
// 	memset(&m_param_trains, 0, sizeof(struct sched_param));

// 	pthread_t m_pThread_trian;
// 	pthread_t m_pThread_dsst;

// 	pthread_attr_init(&m_attr_trains);
// 	pthread_attr_getinheritsched(&m_attr_trains, &m_inher_trains);

// 	if (m_inher_trains == PTHREAD_INHERIT_SCHED)
// 	{
// 		m_inher_trains = PTHREAD_EXPLICIT_SCHED;
// 	}

// 	pthread_attr_setinheritsched(&m_attr_trains, m_inher_trains);
// 	pthread_attr_setschedpolicy(&m_attr_trains, SCHED_FIFO);
// 	m_param_trains.sched_priority = 50;
// 	pthread_attr_setschedparam(&m_attr_trains, &m_param_trains);

// 	pthread_create(&m_pThread_trian, &m_attr_trains, DealProc, this);
// 	pthread_create(&m_pThread_dsst, &m_attr_trains, DealProc2, this);

// #endif
// 	// nothing
// }
TrackerXFDImpl::TrackerXFDImpl(const TrackerXFDImpl::Params &parameters) :
    params(parameters)
{
    // nothing
    update_flag = 0;
    interrupt_flag = 0;
}

void TrackerXFDImpl::setInitialMask(InputArray mask)
{
    preset_mask = mask.getMat();
}

bool TrackerXFDImpl::check_mask_area(const Mat &mat, const double obj_area)
{
    double threshold = 0.05;
    double mask_area= sum(mat)[0];
    if(mask_area < threshold*obj_area) {
        return false;
    }
    return true;
}

Mat TrackerXFDImpl::calculate_response(const Mat &image, const std::vector<Mat> filter)
{
    Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),cvFloor(current_scale_factor * template_size.height));
	int64 t1 = cv::getTickCount();
	resize(patch, patch, rescaled_template_size, 0, 0, INTER_CUBIC);
	int64 t2 = cv::getTickCount();
	double tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	std::cout << "pky: time_resize2 = " << tick_counter << std::endl;
	std::cout << "pky: size_resize2_patch = " << patch.size() << " rescaled_template_size = " << rescaled_template_size <<  "yf_size = " << yf.size() << std::endl;

    std::vector<Mat> ftrs = get_features(patch, yf.size());
    std::vector<Mat> Ffeatures = fourier_transform_features(ftrs);
    Mat resp, res;
    if(params.use_channel_weights)
	{
        res = Mat::zeros(Ffeatures[0].size(), CV_32FC2);
        Mat resp_ch;
        Mat mul_mat;
        for(size_t i = 0; i < Ffeatures.size(); ++i) 
		{
            mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0, true);
            res += (resp_ch * filter_weights[i]);
        }
        idft(res, res, DFT_SCALE | DFT_REAL_OUTPUT);
    } 
	else 
	{
        res = Mat::zeros(Ffeatures[0].size(), CV_32FC2);
        Mat resp_ch;
		int64 t1 = cv::getTickCount();
		for(size_t i = 0; i < Ffeatures.size(); ++i) 
		{
			//std::cout << "pky:  Ffeatures.size()  " << i << "=" << Ffeatures[i].rows << "*" << Ffeatures[i].cols << std::endl;
			//std::cout << "pky:  filter.size()  " << i << "=" << filter[i].rows << "*" << filter[i].cols << std::endl;
            mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0 , true);
            res = res + resp_ch;
        }
		int64 t2 = cv::getTickCount();
		double tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
		std::cout << "pky: time_update_mulspec1 = " << tick_counter << std::endl;
		t1 = cv::getTickCount();
        idft(res, res, DFT_SCALE | DFT_REAL_OUTPUT);
		t2 = cv::getTickCount();
		tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
		std::cout << "pky: time_idft_calculateresponse = " << tick_counter << std::endl;
    }

    return res;
}

void TrackerXFDImpl::update_xfd_filter(const Mat &image, const Mat &mask)
{
	std::cout << "update xfd filter" << std::endl;
    Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),cvFloor(current_scale_factor * template_size.height));
    resize(patch, patch, rescaled_template_size, 0, 0, INTER_CUBIC);

	
    std::vector<Mat> ftrs = get_features(patch, yf.size());
    std::vector<Mat> Fftrs = fourier_transform_features(ftrs);
    std::vector<Mat> new_xfd_filter = create_xfd_filter(Fftrs, yf, mask);
	std::cout << "patch_uxfd = " << patch.size() << "rescaled_template_size = " << rescaled_template_size << "yf = "<< yf.size() << "ftrs = "<< ftrs.size() << std::endl;
    //calculate per channel weights
    if(params.use_channel_weights) 
	{
        Mat current_resp;
        double max_val;
        float sum_weights = 0;
        std::vector<float> new_filter_weights = std::vector<float>(new_xfd_filter.size());
        for(size_t i = 0; i < new_xfd_filter.size(); ++i) 
		{
            mulSpectrums(Fftrs[i], new_xfd_filter[i], current_resp, 0, true);
            idft(current_resp, current_resp, DFT_SCALE | DFT_REAL_OUTPUT);
            minMaxLoc(current_resp, NULL, &max_val, NULL, NULL);
            sum_weights += static_cast<float>(max_val);
            new_filter_weights[i] = static_cast<float>(max_val);
        }
        //update filter weights with new values
        float updated_sum = 0;
        for(size_t i = 0; i < filter_weights.size(); ++i) 
		{
            filter_weights[i] = filter_weights[i]*(1.0f - params.weights_lr) +
                params.weights_lr * (new_filter_weights[i] / sum_weights);
            updated_sum += filter_weights[i];
        }
        //normalize weights
        for(size_t i = 0; i < filter_weights.size(); ++i) 
		{
            filter_weights[i] /= updated_sum;
        }
    }
    for(size_t i = 0; i < xfd_filter.size(); ++i) 
	{
        xfd_filter[i] = (1.0f - params.filter_lr)*xfd_filter[i] + params.filter_lr * new_xfd_filter[i];
    }
	std::cout << "xfd_filter.size()" << xfd_filter.size() << std::endl;
    std::vector<Mat>().swap(ftrs);
    std::vector<Mat>().swap(Fftrs);
}

void saveMatToFile(const cv::Mat& matrix, const std::string& filename) {
	std::ofstream file(filename);
	if (file.is_open()) {
		// 逐行保存矩阵数据
		for (int i = 0; i < matrix.rows; ++i) {
			for (int j = 0; j < matrix.cols; ++j) {
				file << matrix.at<float>(i, j) << " ";
			}
			file << std::endl;
		}

		file.close();
		std::cout << "矩阵数据已成功保存到文件：" << filename << std::endl;
	}
	else {
		std::cerr << "无法打开文件：" << filename << std::endl;
	}
}

std::vector<Mat> TrackerXFDImpl::get_features(const Mat &patch, const Size2i &feature_size)
{
    std::vector<Mat> features;
    if (params.use_hog) {
        std::vector<Mat> hog = get_features_hog(patch, cell_size);
		// Size targetSize = hog[0].size();
        features.insert(features.end(), hog.begin(),hog.begin() + params.num_hog_channels_used);
    }
	// std::vector<Mat> features;
	// std::vector<Mat> hog = get_features_hog(patch, cell_size);
	// Size targetSize = hog[0].size();
	// features.insert(features.end(), hog.begin(),hog.begin() + params.num_hog_channels_used);

	// std::vector<Mat> features_cnn = get_features_cnn(patch, cell_size);
	// std::cout <<  "pky:feature_cnn_size()" << features_cnn.size() << " and " << features_cnn[0].size() << std::endl;
	// //std::cout <<  "pky:features_size()" << features.size() << " and " << features[0].size() <<std::endl;
	// //Size targetSize = features[0].size();
	// for (size_t i = 0; i < features_cnn.size(); ++i) {
	// 	resize(features_cnn[i], features_cnn[i], targetSize);
	// }
	// features.insert(features.end(), features_cnn.begin(), features_cnn.begin() + params.num_cnn_channels_used);
	// std::cout <<  "pky:feature_cnn_size end " << std::endl;

	// if (params.use_cnn) {
	// 	std::vector<Mat> features_cnn = get_features_cnn(patch, cell_size);
	// 	std::cout <<  "pky:feature_cnn_size()" << features_cnn.size() << " and " << features_cnn[0].size() << std::endl;
    //     // 生成文件名
    //     static int current_frame = 0;
    //     std::string filename1 = std::to_string(current_frame++) + "matrix1.txt";
    //     std::string filename2 = std::to_string(current_frame++) + "matrix2.txt";
	// 	//std::cout <<  "pky:features_size()" << features.size() << " and " << features[0].size() <<std::endl;
	// 	Size targetSize = features[0].size();
	// 	for (size_t i = 0; i < features_cnn.size(); ++i) {
    //     	resize(features_cnn[i], features_cnn[i], targetSize);
    //         int i0,j0;
    //         if(i == 0)
    //         {
    //             std::ofstream file1(filename1);
    //             if(file1.is_open())
    //             {
    //                 file1 << "Matrix 1 (56*56):\n";
    //                 for(i0 = 0;i0 < features_cnn[i].rows;i0++)
    //                 {
    //                     for(j0 = 0;j0 < features_cnn[i].cols;j0++)
    //                     {
    //                         file1 << features_cnn[i].at<float>(i0,j0) << " ";
    //                     }
    //                     file1 << "\n";
    //                 }
    //                 file1.close();
    //                 std::cout <<  "pky:features111111111111111111111" <<std::endl;
    //             }
    //             else
    //             {
    //                 std::cerr << "Error opening file for Matrix1\n";
    //             }
    //         }
    //         if(i == 64)
    //         {
    //             std::ofstream file2(filename2);
    //             if(file2.is_open())
    //             {
    //                 file2 << "Matrix 2 (56*56):\n";
    //                 for(i0 = 0;i0 < features_cnn[i].rows;i0++)
    //                 {
    //                     for(j0 = 0;j0 < features_cnn[i].cols;j0++)
    //                     {
    //                         file2 << features_cnn[i].at<float>(i0,j0) << " ";
    //                     }
    //                     file2 << "\n";
    //                 }
    //                 file2.close();
    //                 std::cout <<  "pky:features22222222222222222222" <<std::endl;
    //             }
    //             else
    //             {
    //                 std::cerr << "Error opening file for Matrix2\n";
    //             }
    //         }
    // 	}
	// 	// features.insert(features.end(), features_cnn.begin(), features_cnn.begin() + params.num_cnn_channels_used);
	// 	std::cout <<  "pky:feature_cnn_size end " << std::endl;
	// }
    if (params.use_color_names) {
        std::vector<Mat> cn;
        cn = get_features_cn(patch, feature_size);
        features.insert(features.end(), cn.begin(), cn.end());
    }
    if(params.use_gray) {
        Mat gray_m;
        cvtColor(patch, gray_m, COLOR_BGR2GRAY);
        resize(gray_m, gray_m, feature_size, 0, 0, INTER_CUBIC);
        gray_m.convertTo(gray_m, CV_32FC1, 1.0/255.0, -0.5);
        features.push_back(gray_m);
    }
    if(params.use_rgb) {
        std::vector<Mat> rgb_features = get_features_rgb(patch, feature_size);
        features.insert(features.end(), rgb_features.begin(), rgb_features.end());
    }

    for (size_t i = 0; i < features.size(); ++i) {
        features.at(i) = features.at(i).mul(window);
    }
    return features;
}

class ParallelCreateXFDFilter : public ParallelLoopBody {
public:
	ParallelCreateXFDFilter(
        const std::vector<cv::Mat> img_features,
        const cv::Mat Y,
        const cv::Mat P,
        int admm_iterations,
        std::vector<Mat> &result_filter_):
        result_filter(result_filter_)
    {
        this->img_features = img_features;
        this->Y = Y;
        this->P = P;
        this->admm_iterations = admm_iterations;
    }
    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
		int64 t1 = cv::getTickCount();
        for (int i = range.start; i < range.end; i++) {
            float mu = 5.0f;
            float beta = 3.0f;
            float mu_max = 20.0f;
            float lambda = mu / 100.0f;

            Mat F = img_features[i];
			int64 t10 = cv::getTickCount();
            Mat Sxy, Sxx;
            mulSpectrums(F, Y, Sxy, 0, true);
            mulSpectrums(F, F, Sxx, 0, true);

            Mat H;
			int64 t100 = cv::getTickCount();
            H = divide_complex_matrices(Sxy, (Sxx + lambda));
			int64 t200 = cv::getTickCount();
			double tick_counter10 = (t200 - t100) * 1000.0 / cv::getTickFrequency();
			//std::cout << "pky: time_Paralleldivide_complex_matrices = " << tick_counter10 << std::endl;
            idft(H, H, DFT_SCALE|DFT_REAL_OUTPUT);
			t100 = cv::getTickCount();
            H = H.mul(P);
			t200 = cv::getTickCount();
			tick_counter10 = (t200 - t100) * 1000.0 / cv::getTickFrequency();
			//std::cout << "pky: time_Parallelmulmat = " << tick_counter10 << std::endl;
			idft(H, H, DFT_SCALE | DFT_REAL_OUTPUT);
            dft(H, H, DFT_COMPLEX_OUTPUT);
			int64 t20 = cv::getTickCount();
			double tick_counter1 = (t20 - t10) * 1000.0 / cv::getTickFrequency();
			//std::cout << "pky: time_ParallelCreateXFDFilter_apart = " << tick_counter1 << std::endl;
            Mat L = Mat::zeros(H.size(), H.type()); //Lagrangian multiplier
            Mat G;
			t10 = cv::getTickCount();
            for(int iteration = 0; iteration < admm_iterations; ++iteration) {
                G = divide_complex_matrices((Sxy + (mu * H) - L) , (Sxx + mu));
                idft((mu * G) + L, H, DFT_SCALE | DFT_REAL_OUTPUT);
                float lm = 1.0f / (lambda+mu);
                H = H.mul(P*lm);
                dft(H, H, DFT_COMPLEX_OUTPUT);

                //Update variables for next iteration
                L = L + mu * (G - H);
                mu = min(mu_max, beta*mu);
            }
			t20 = cv::getTickCount();
			tick_counter1 = (t20 - t10) * 1000.0 / cv::getTickFrequency();
			//std::cout << "pky: time_ParallelCreateXFDFilter_admm = " << tick_counter1 << std::endl;
            result_filter[i] = H;
        }
		int64 t2 = cv::getTickCount();
		double tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
		std::cout << "pky: time_ParallelCreateXFDFilter = " << tick_counter << std::endl;
    }

	ParallelCreateXFDFilter& operator=(const ParallelCreateXFDFilter &) {
        return *this;
    }

private:
    int admm_iterations;
    Mat Y;
    Mat P;
    std::vector<Mat> img_features;
    std::vector<Mat> &result_filter;
};


std::vector<Mat> TrackerXFDImpl::create_xfd_filter(
        const std::vector<cv::Mat> img_features,
        const cv::Mat Y,
        const cv::Mat P)
{
    std::vector<Mat> result_filter;
    result_filter.resize(img_features.size());
	ParallelCreateXFDFilter parallelCreateXFDFilter(img_features, Y, P,
            params.admm_iterations, result_filter);
    parallel_for_(Range(0, static_cast<int>(result_filter.size())), parallelCreateXFDFilter);
	std::cout << "ParallelCreateXFDFilter" << std::endl;

    return result_filter;
}

Mat TrackerXFDImpl::get_location_prior(
        const Rect roi,
        const Size2f target_size,
        const Size img_sz)
{
    int x1 = cvRound(max(min(roi.x-1, img_sz.width-1) , 0));
    int y1 = cvRound(max(min(roi.y-1, img_sz.height-1) , 0));

    int x2 = cvRound(min(max(roi.width-1, 0) , img_sz.width-1));
    int y2 = cvRound(min(max(roi.height-1, 0) , img_sz.height-1));

    Size target_sz;
    target_sz.width = target_sz.height = cvFloor(min(target_size.width, target_size.height));

    double cx = x1 + (x2-x1)/2.;
    double cy = y1 + (y2-y1)/2.;
    double kernel_size_width = 1.0/(0.5*static_cast<double>(target_sz.width)*1.4142+1);
    double kernel_size_height = 1.0/(0.5*static_cast<double>(target_sz.height)*1.4142+1);

    cv::Mat kernel_weight = Mat::zeros(1 + cvFloor(y2 - y1) , 1+cvFloor(-(x1-cx) + (x2-cx)), CV_64FC1);
    for (int y = y1; y < y2+1; ++y){
        double * weightPtr = kernel_weight.ptr<double>(y);
        double tmp_y = std::pow((cy-y)*kernel_size_height, 2);
        for (int x = x1; x < x2+1; ++x){
            weightPtr[x] = kernel_epan(std::pow((cx-x)*kernel_size_width,2) + tmp_y);
        }
    }

    double max_val;
    cv::minMaxLoc(kernel_weight, NULL, &max_val, NULL, NULL);
    Mat fg_prior = kernel_weight / max_val;
    fg_prior.setTo(0.5, fg_prior < 0.5);
    fg_prior.setTo(0.9, fg_prior > 0.9);
    return fg_prior;
}

Mat TrackerXFDImpl::segment_region(
        const Mat &image,
        const Point2f &object_center,
        const Size2f &template_size,
        const Size &target_size,
        float scale_factor)
{
    Rect valid_pixels;
    Mat patch = get_subwindow(image, object_center, cvFloor(scale_factor * template_size.width),
        cvFloor(scale_factor * template_size.height), &valid_pixels);
    Size2f scaled_target = Size2f(target_size.width * scale_factor,
            target_size.height * scale_factor);
    Mat fg_prior = get_location_prior(
            Rect(0,0, patch.size().width, patch.size().height),
            scaled_target , patch.size());

    std::vector<Mat> img_channels;
    split(patch, img_channels);
    std::pair<Mat, Mat> probs = Segment::computePosteriors2(img_channels, 0, 0, patch.cols, patch.rows,
                    p_b, fg_prior, 1.0-fg_prior, hist_foreground, hist_background);

    Mat mask = Mat::zeros(probs.first.size(), probs.first.type());
    probs.first(valid_pixels).copyTo(mask(valid_pixels));
    double max_resp = get_max(mask);
    threshold(mask, mask, max_resp / 2.0, 1, THRESH_BINARY);
    mask.convertTo(mask, CV_32FC1, 1.0);

    return mask;
}


void TrackerXFDImpl::extract_histograms(const Mat &image, cv::Rect region, Histogram &hf, Histogram &hb)
{
    // get coordinates of the region
    int x1 = std::min(std::max(0, region.x), image.cols-1);
    int y1 = std::min(std::max(0, region.y), image.rows-1);
    int x2 = std::min(std::max(0, region.x + region.width), image.cols-1);
    int y2 = std::min(std::max(0, region.y + region.height), image.rows-1);

    // calculate coordinates of the background region
    int offsetX = (x2-x1+1) / params.background_ratio;
    int offsetY = (y2-y1+1) / params.background_ratio;
    int outer_y1 = std::max(0, (int)(y1-offsetY));
    int outer_y2 = std::min(image.rows, (int)(y2+offsetY+1));
    int outer_x1 = std::max(0, (int)(x1-offsetX));
    int outer_x2 = std::min(image.cols, (int)(x2+offsetX+1));

    // calculate probability for the background
    p_b = 1.0 - ((x2-x1+1) * (y2-y1+1)) /
        ((double) (outer_x2-outer_x1+1) * (outer_y2-outer_y1+1));

    // split multi-channel image into the std::vector of matrices
    std::vector<Mat> img_channels(image.channels());
    split(image, img_channels);
    for(size_t k=0; k<img_channels.size(); k++) {
        img_channels.at(k).convertTo(img_channels.at(k), CV_8UC1);
    }

    hf.extractForegroundHistogram(img_channels, Mat(), false, x1, y1, x2, y2);
    hb.extractBackGroundHistogram(img_channels, x1, y1, x2, y2,
        outer_x1, outer_y1, outer_x2, outer_y2);
    std::vector<Mat>().swap(img_channels);
}

void TrackerXFDImpl::update_histograms(const Mat &image, const Rect &region)
{
    // create temporary histograms
    Histogram hf(image.channels(), params.histogram_bins);
    Histogram hb(image.channels(), params.histogram_bins);
    extract_histograms(image, region, hf, hb);

    // get histogram vectors from temporary histograms
    std::vector<double> hf_vect_new = hf.getHistogramVector();
    std::vector<double> hb_vect_new = hb.getHistogramVector();
    // get histogram vectors from learned histograms
    std::vector<double> hf_vect = hist_foreground.getHistogramVector();
    std::vector<double> hb_vect = hist_background.getHistogramVector();

    // update histograms - use learning rate
    for(size_t i=0; i<hf_vect.size(); i++) {
        hf_vect_new[i] = (1-params.histogram_lr)*hf_vect[i] +
            params.histogram_lr*hf_vect_new[i];
        hb_vect_new[i] = (1-params.histogram_lr)*hb_vect[i] +
            params.histogram_lr*hb_vect_new[i];
    }

    // set learned histograms
    hist_foreground.setHistogramVector(&hf_vect_new[0]);
    hist_background.setHistogramVector(&hb_vect_new[0]);

    std::vector<double>().swap(hf_vect);
    std::vector<double>().swap(hb_vect);
}

double TrackerXFDImpl::calculate_resp_quality(const Mat &image, Point max_loc , double max_val)
{
	double dnum = 0;

	int x1 = (max_loc.x - (int)(image.cols * 0.05 + 0.5)) > 0 ? (max_loc.x - (int)(image.cols * 0.05 + 0.5)) : 0;
	int x2 = (max_loc.x + (int)(image.cols * 0.05 + 0.5)) < image.cols ? (max_loc.x + (int)(image.cols * 0.05 + 0.5)) : image.cols - 1;
	int y1 = (max_loc.y - (int)(image.rows * 0.05 + 0.5)) > 0 ? (max_loc.y - (int)(image.rows * 0.05 + 0.5)) : 0;
	int y2 = (max_loc.y + (int)(image.rows * 0.05 + 0.5)) < image.rows ? (max_loc.y + (int)(image.rows * 0.05 + 0.5)) : image.rows - 1;//pky:why rows*0.05+0.5
	int64 t1 = cv::getTickCount();
	for (int i = 0; i < image.cols; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			if ((i > x1 && i < x2) && (j > y1 && j < y2)){}
			else
			{
				dnum += image.at<float>(i, j);
			}
		}
	}

	double mu_s = dnum / (image.cols * image.rows);

	dnum = 0;

	for (int i = 0; i < image.cols; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			if ((i > x1 && i < x2) && (j > y1 && j < y2)) {}
			else
			{
				dnum += (image.at<float>(i, j) - mu_s)*(image.at<float>(i, j) - mu_s);
			}
		}
	}
	int64 t2 = cv::getTickCount();
	double tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	std::cout << "pky: time_resp_quality = " << tick_counter << std::endl;
	std::cout << "pky: size_resp_quality = " << image.rows << "*" << image.cols << std::endl;
	
	double sigma_s = sqrt((dnum / (image.cols * image.rows)));
	double s = max_val * (max_val - mu_s) / (sigma_s + 0.0001);

	return s;
}

void TrackerXFDImpl::calculate_resp_score(double resp_quality)
{
	if (Resp_num == 0)
	{
		resp_quality_init = resp_quality;
		resp_norm = 1;
		resp_mean = 1;
		Resp_score = 0;
	}
	else
	{
		resp_norm = resp_quality / resp_quality_init;

		if (Resp_score < 2.0)
		{
			resp_mean = (resp_mean * Resp_num + resp_norm) / (Resp_num + 1);
		}

		if (Resp_num > 5)
		{
			Resp_score = (resp_mean - resp_norm) / resp_norm;
		}
	}

	Resp_num++;
	if (Resp_num > 200000)
	{
		Resp_num = 200000;
	}

}

bool TrackerXFDImpl::judge_loss_Target(double response_max,double response_score,double secondmax_response)
{
	bool ret = false;
	float now_psr_th = params.psr_threshold;

	printf("response_max = %.6lf, now_psr_th = %.6lf\n", response_max, now_psr_th);
	printf("response_score = %.6lf, params.score_threshold = %.6lf\n", response_score, params.score_threshold);


	//判断位置估计质量
	if (response_max < now_psr_th) // target "lost" //单次响应过低
	{
		printf("target lost 1 :%.6lf < %.6lf \n", response_max, now_psr_th);
		flag_loss_target = true;
	}
	else if (response_score > params.score_threshold)// target "lost" //跟踪质量不足
	{
		printf("target lost 2: %.6lf > %.6lf \n", response_score, params.score_threshold);
		flag_loss_target = true;
	}
	else
	{
		flag_loss_target = false;
	}


	////判断是否更新模板
	//if (response_max < now_psr_th * 0.5) // target "lost" //单次响应过低
	//{
	//	printf("no update 1: %.6lf < %.6lf \n", response_max, now_psr_th);
	//	update_flag = false;
	//}
	//else if (response_score > params.score_threshold / 2.0)// target "lost" //跟踪质量不足
	//{
	//	printf("no update 2: %.6lf < %.6lf \n", response_score, params.score_threshold);
	//	update_flag = false;
	//}
	//else
	//{
	//	update_flag = true;
	//}


	update_flag = true;

	//else if (secondmax_response > 0.6*response_max)// target "lost" 相似目标
	//{
	//	printf("target lost : Similar Target!\n");
	//	ret = true;
	//}

	return ret;
}


int TrackerXFDImpl::imregionalmax(Mat input, int num_LocMax, float threshold, int minDistBtwLocMax, 
	int assignSecondNum, double &assignSecondMaximum)
{
	Mat scratch = input.clone();
	int nFoundLocMax = 0;
	assignSecondMaximum = 0.0;

	for (int i = 0; i < num_LocMax; i++)
	{
		Point location;
		double maxVal;
		minMaxLoc(scratch, NULL, &maxVal, NULL, &location);
		 
		if (maxVal > threshold) 
		{
			nFoundLocMax += 1;
			if (nFoundLocMax == assignSecondNum)
			{
				assignSecondMaximum = maxVal;
			}

			int row = location.y;
			int col = location.x;

			int r0 = (row - minDistBtwLocMax > -1 ? row - minDistBtwLocMax : 0);
			int r1 = (row + minDistBtwLocMax < scratch.rows ? row + minDistBtwLocMax : scratch.rows - 1);
			int c0 = (col - minDistBtwLocMax > -1 ? col - minDistBtwLocMax : 0);
			int c1 = (col + minDistBtwLocMax < scratch.cols ? col + minDistBtwLocMax : scratch.cols - 1);
			for (int r = r0; r <= r1; r++) 
			{
				for (int c = c0; c <= c1; c++) 
				{
					//if (vdist(Point2DMake(r, c), Point2DMake(row, col)) <= minDistBtwLocMax) {
					if ( sqrtf( (float)((r - row)*(r - row) + (c - col)*(c - col))) <= (float)minDistBtwLocMax)
					{
						scratch.at<float>(r, c) = 0.0;
					}
				}
			}
		}
		else 
		{
			break;
		}
	}
	return nFoundLocMax;
}


Point2f TrackerXFDImpl::estimate_new_position(const Mat &image)
{
	Point2f new_center = object_center;
	int64 t1 = cv::getTickCount();
    Mat resp = calculate_response(image, xfd_filter);
	int64 t2 = cv::getTickCount();
	double tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	std::cout << "pky: time_calculate_response_u = " << tick_counter << std::endl;

	double max_val;
	Point max_loc;
	t1 = cv::getTickCount();
	minMaxLoc(resp, NULL, &max_val, NULL, &max_loc);
	t2 = cv::getTickCount();
	tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	std::cout << "pky: time_minMaxLoc = " << tick_counter << std::endl;
	
	double n_assignSecondMaximum = 0.0;
	int	n_assignSecondNum = 6;
	int	n_minDistBtwLocMax = 2;

	//imregionalmax(resp, n_assignSecondNum + 1, params.psr_threshold, n_minDistBtwLocMax, n_assignSecondNum, n_assignSecondMaximum);
	t1 = cv::getTickCount();
	double n_resp_quality = calculate_resp_quality(resp, max_loc, max_val);

	calculate_resp_score(n_resp_quality);
	judge_loss_Target(max_val, Resp_score, n_assignSecondMaximum);//pky:so second=0?
	t2 = cv::getTickCount();
	tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	std::cout << "pky: time_score = " << tick_counter << std::endl;

	//if (flag_loss_target == true)
	//{
	//	return new_center;
	//}

    // take into account also subpixel accuracy
    float col = ((float) max_loc.x) + subpixel_peak(resp, "horizontal", max_loc);
    float row = ((float) max_loc.y) + subpixel_peak(resp, "vertical", max_loc);
    if(row + 1 > (float)resp.rows / 2.0f) {
        row = row - resp.rows;
    }
    if(col + 1 > (float)resp.cols / 2.0f) {
        col = col - resp.cols;
    }

    // calculate x and y displacements
    new_center = object_center + Point2f(current_scale_factor * (1.0f / rescale_ratio) *cell_size*(col),current_scale_factor * (1.0f / rescale_ratio) *cell_size*(row));
    
    if(new_center.x < 0)
    {   
		new_center.x = 0;
	}
    if(new_center.x >= image_size.width)
    {    
		new_center.x = static_cast<float>(image_size.width - 1);
	}
    if(new_center.y < 0)
    {    
		new_center.y = 0;
	}
    if(new_center.y >= image_size.height)
    {    
		new_center.y = static_cast<float>(image_size.height - 1);
	}

	
    return new_center;
}

// #ifdef WIN_X64
// DWORD WINAPI TrackerXFDImpl::DealProc(LPVOID lpParam)
// {
// 	TrackerXFDImpl * p_func = (TrackerXFDImpl *)lpParam;

// 	while (1)
// 	{
// 		if (p_func->update_flag == 0)
// 		{
// 			Sleep(10);
// 			continue;
// 		}

// 		int64 t1 = cv::getTickCount();
		
// 		EnterCriticalSection(&p_func->g_cs);
// 		Mat img_loc = p_func->img_now.clone();
// 		LeaveCriticalSection(&p_func->g_cs);

// 		if (p_func->interrupt_flag == 1) { p_func->update_flag = 0;   continue; }

// 		//update tracker
// 		if (p_func->params.use_segmentation)
// 		{
// 			//Mat hsv_img = bgr2hsv(image);
// 			//update_histograms(hsv_img, bounding_box);
// 			//filter_mask = segment_region(hsv_img, object_center,
// 			//        template_size,original_target_size, current_scale_factor);

// 			p_func->update_histograms(img_loc, p_func->bounding_box);
// 			p_func->filter_mask = p_func->segment_region(img_loc, p_func->object_center,
// 				p_func->template_size, p_func->original_target_size, p_func->current_scale_factor);

// 			resize(p_func->filter_mask, p_func->filter_mask, p_func->yf.size(), 0, 0, INTER_NEAREST);
			
// 			if (p_func->check_mask_area(p_func->filter_mask, p_func->default_mask_area)) 
// 			{
// 				dilate(p_func->filter_mask, p_func->filter_mask, p_func->erode_element);
// 			}
// 			else 
// 			{
// 				p_func->filter_mask = p_func->default_mask;
// 			}

			
// 		}
// 		else 
// 		{
// 			p_func->filter_mask = p_func->default_mask;
// 		}

		
// 		if (p_func->interrupt_flag == 1) { p_func->update_flag = 0;  continue; }

// 		p_func->update_xfd_filter(img_loc, p_func->filter_mask);

// 		if (p_func->interrupt_flag == 1) { p_func->update_flag = 0;  continue; }

// 		p_func->update_flag = 0;
// 		int64 t2 = cv::getTickCount();
// 		double during = static_cast<double>(t2 - t1)* 1000.0 / cv::getTickFrequency();
// 		//std::cout << "filter: " << during << " ms " << std::endl;

// 	}

// 	return 0;
// }
// #else
// void * TrackerXFDImpl::DealProc(void * lpParam)
// {
// 	cpu_set_t m_mask;
// 	CPU_ZERO(&m_mask);
// 	CPU_SET(6, &m_mask);

// 	if (pthread_setaffinity_np(pthread_self(), sizeof(m_mask), &m_mask) < 0)
// 	{
// 		//printf("set thread affinity failed\n");
// 	}

// 	TrackerXFDImpl * p_func = (TrackerXFDImpl *)lpParam;

// 	struct timeval m_start_time;
// 	struct timeval m_end_time;
//     Mat img_loc;

// 	while (1)
// 	{

// 		if (p_func->update_flag == 0)
// 		{
// 			usleep(10*1000);
// 			continue;
// 		}

//         int64 t1 = cv::getTickCount();
		
// 		pthread_mutex_lock(&p_func->m_mutex_tracker);
//         if(!p_func->img_now.empty())
//         {
//            img_loc = p_func->img_now.clone();
//         }
//         else
//         { 
//             p_func->update_flag = 0;  
//             continue; 
//         }
// 		pthread_mutex_unlock(&p_func->m_mutex_tracker);

// 		if (p_func->interrupt_flag == 1) { p_func->update_flag = 0;  continue; }

// 		//update tracker
// 		if (p_func->params.use_segmentation)
// 		{
// 			//Mat hsv_img = bgr2hsv(image);
// 			//update_histograms(hsv_img, bounding_box);
// 			//filter_mask = segment_region(hsv_img, object_center,
// 			//        template_size,original_target_size, current_scale_factor);

// 			p_func->update_histograms(img_loc, p_func->bounding_box);
// 			p_func->filter_mask = p_func->segment_region(img_loc, p_func->object_center,
// 				p_func->template_size, p_func->original_target_size, p_func->current_scale_factor);

// 			resize(p_func->filter_mask, p_func->filter_mask, p_func->yf.size(), 0, 0, INTER_NEAREST);
// 			if (p_func->check_mask_area(p_func->filter_mask, p_func->default_mask_area)) {
// 				dilate(p_func->filter_mask, p_func->filter_mask, p_func->erode_element);
// 			}
// 			else {
// 				p_func->filter_mask = p_func->default_mask;
// 			}
// 		}
// 		else {
// 			p_func->filter_mask = p_func->default_mask;
// 		}

// 		if (p_func->interrupt_flag == 1) { p_func->update_flag = 0;  continue; }

// 		p_func->update_xfd_filter(img_loc, p_func->filter_mask);

// 		if (p_func->interrupt_flag == 1) { p_func->update_flag = 0;   continue; }

// 		p_func->update_flag = 0;

// 		int64 t2 = cv::getTickCount();
// 		double during = static_cast<double>(t2 - t1)* 1000.0 / cv::getTickFrequency();
// 		std::cout << "time_filter: " << during << " ms " << std::endl;

// 	}

// 	return NULL;
// }
// #endif

// #ifdef WIN_X64
// DWORD WINAPI TrackerXFDImpl::DealProc2(LPVOID lpParam)
// {
// 	TrackerXFDImpl * p_func = (TrackerXFDImpl *)lpParam;

// 	while (1)
// 	{
// 		if (p_func->dsst_flag == 0)
// 		{
// 			Sleep(10);
// 			continue;
// 		}

// 		int64 t1 = cv::getTickCount();

// 		EnterCriticalSection(&p_func->g_cs);
// 		Mat img_loc = p_func->img_now.clone();
// 		LeaveCriticalSection(&p_func->g_cs);

// 		if (p_func->interrupt2_flag == 1) { p_func->dsst_flag = 0;   continue; }
		
// 		p_func->current_scale_factor = p_func->dsst.getScale(img_loc, p_func->object_center);

// 		if (p_func->interrupt2_flag == 1) { p_func->dsst_flag = 0;   continue; }

// 		p_func->dsst.update(img_loc, p_func->object_center);//update bouding_box according to new scale and location

// 		p_func->dsst_flag = 0;
// 	}

// 	return 0;
// }
// #else
// void * TrackerXFDImpl::DealProc2(void * lpParam)
// {
// 	cpu_set_t m_mask;
// 	CPU_ZERO(&m_mask);
// 	CPU_SET(7, &m_mask);

// 	if (pthread_setaffinity_np(pthread_self(), sizeof(m_mask), &m_mask) < 0)
// 	{
// 		//printf("set thread affinity failed\n");
// 	}

// 	TrackerXFDImpl * p_func = (TrackerXFDImpl *)lpParam;

// 	while (1)
// 	{
// 		if (p_func->dsst_flag == 0)
// 		{
// 			usleep(10*1000);
// 			continue;
// 		}

// 		int64 t1 = cv::getTickCount();
		
// 		pthread_mutex_lock(&p_func->m_mutex_tracker);
// 		Mat img_loc = p_func->img_now.clone();
// 		pthread_mutex_unlock(&p_func->m_mutex_tracker);

// 		if (p_func->interrupt2_flag == 1) { p_func->dsst_flag = 0;   continue; }

// 		p_func->current_scale_factor = p_func->dsst.getScale(img_loc, p_func->object_center);

// 		if (p_func->interrupt2_flag == 1) { p_func->dsst_flag = 0;   continue; }

// 		p_func->dsst.update(img_loc, p_func->object_center);//update bouding_box according to new scale and location

// 		p_func->dsst_flag = 0;
// 		int64 t2 = cv::getTickCount();
// 		double during = static_cast<double>(t2 - t1)* 1000.0 / cv::getTickFrequency();
// 		std::cout << "time_chidu: " << during << " ms " << std::endl;

// 	}

// 	return NULL;
// }
// #endif


// *********************************************************************
// *                        Update API function                        *
// *********************************************************************
// bool TrackerXFDImpl::update(InputArray image_, Rect& boundingBox)
// {
//     Mat image;
// 	if (image_.channels() == 1)    //treat gray image as color image
// 	{
// 		cvtColor(image_, image, COLOR_GRAY2BGR);
// 	}
// 	else
// 	{
// 		image = image_.getMat();
// 	}

// 	int64 t1 = cv::getTickCount();
// 	object_center = estimate_new_position(image);
// 	int64 t2 = cv::getTickCount();
// 	double during = static_cast<double>(t2 - t1)* 1000.0 / cv::getTickFrequency();
// 	std::cout << "time_position: " << during << " ms " << std::endl;




// #ifdef WIN_X64

// 	EnterCriticalSection(&g_cs);
// 	bounding_box.x = object_center.x - current_scale_factor * original_target_size.width / 2.0f;
// 	bounding_box.y = object_center.y - current_scale_factor * original_target_size.height / 2.0f;
// 	bounding_box.width = current_scale_factor * original_target_size.width;
// 	bounding_box.height = current_scale_factor * original_target_size.height;
// 	img_now = image.clone();
// 	LeaveCriticalSection(&g_cs);

// 	dsst_flag = 1;

// #else

// 	pthread_mutex_lock(&m_mutex_tracker);
// 	bounding_box.x = object_center.x - current_scale_factor * original_target_size.width / 2.0f;
// 	bounding_box.y = object_center.y - current_scale_factor * original_target_size.height / 2.0f;
// 	bounding_box.width = current_scale_factor * original_target_size.width;
// 	bounding_box.height = current_scale_factor * original_target_size.height;
// 	img_now = image.clone();
// 	pthread_mutex_unlock(&m_mutex_tracker);

// 	dsst_flag = 1;

// #endif

// 	boundingBox = bounding_box;

// 	if (flag_loss_target == false)
// 	{
// 		update_flag = 1;
// 	}
// 	else
// 	{
// 		update_flag = 0;
// 	}


//     return true;



// }
bool TrackerXFDImpl::update(InputArray image_, Rect& boundingBox)
{
    Mat image;
    if(image_.channels() == 1)    //treat gray image as color image
        cvtColor(image_, image, COLOR_GRAY2BGR);
    else
        image = image_.getMat();

    object_center = estimate_new_position(image);
    if (object_center.x < 0 && object_center.y < 0)
        return false;

    current_scale_factor = dsst.getScale(image, object_center);
    //update bouding_box according to new scale and location
    bounding_box.x = object_center.x - current_scale_factor * original_target_size.width / 2.0f;
    bounding_box.y = object_center.y - current_scale_factor * original_target_size.height / 2.0f;
    bounding_box.width = current_scale_factor * original_target_size.width;
    bounding_box.height = current_scale_factor * original_target_size.height;

    //update tracker
    if(params.use_segmentation) {
        Mat hsv_img = bgr2hsv(image);
        update_histograms(hsv_img, bounding_box);
        filter_mask = segment_region(hsv_img, object_center,
                template_size,original_target_size, current_scale_factor);
        resize(filter_mask, filter_mask, yf.size(), 0, 0, INTER_NEAREST);
        if(check_mask_area(filter_mask, default_mask_area)) {
            dilate(filter_mask , filter_mask, erode_element);
        } else {
            filter_mask = default_mask;
        }
    } else {
        filter_mask = default_mask;
    }
    update_xfd_filter(image, filter_mask);
    dsst.update(image, object_center);
    boundingBox = bounding_box;
    return true;
}


// *********************************************************************
// *                        Init API function                          *
// *********************************************************************
void TrackerXFDImpl::init(InputArray image_, const Rect& boundingBox)
{
	interrupt_flag = 1;
	Resp_num = 0;
	resp_quality_init = 0.0;
	flag_loss_target = false;
    std::cout << "init 1"  << std::endl;
	while (1)
	{
		if (update_flag == 0)
		{
			interrupt_flag = 0;
			break;
		}
		else
		{

#ifdef WIN_X64
			Sleep(10);
#else
			usleep(10 * 1000);
#endif
			
		}
	}

    Mat image;
	if (image_.channels() == 1)    //treat gray image as color image
	{
		cvtColor(image_, image, COLOR_GRAY2BGR);
	}   
	else
	{
		image = image_.getMat();
	}
    
	if (boundingBox.x <= 0 || boundingBox.y <= 0 || boundingBox.width <= 0 || boundingBox.height <= 0 ||
		boundingBox.x + boundingBox.width >= image.cols - 1 || 
		boundingBox.y + boundingBox.height >= image.rows - 1 )
	{
		flag_loss_target = true;
		//return;
	}
    std::cout << "init 2"  << std::endl;
	update_flag = 0;

    current_scale_factor = 1.0;
    image_size = image.size();
    bounding_box = boundingBox;

	if (bounding_box.x < 0)
	{
		bounding_box.x = 0;
	}
	if (bounding_box.x + bounding_box.width >= image.cols - 1)
	{
		bounding_box.x = image.cols - 1 - bounding_box.width;
	}
	if (bounding_box.y < 0)
	{
		bounding_box.y = 0;
	}
	if (bounding_box.y + bounding_box.height >= image.rows - 1)
	{
		bounding_box.y = image.rows - 1 - bounding_box.height;
	}
std::cout << "init 3"  << std::endl;
	//cell_size = cvFloor(std::min(4.0, std::max(1.0, static_cast<double>(
	//	cvCeil((bounding_box.width * bounding_box.height) / 400.0)))));
	cell_size = 4;

    original_target_size = Size(bounding_box.size());

    template_size.width = static_cast<float>(cvFloor(original_target_size.width + params.padding *
            sqrt(original_target_size.width * original_target_size.height)));
    template_size.height = static_cast<float>(cvFloor(original_target_size.height + params.padding *
            sqrt(original_target_size.width * original_target_size.height)));
    template_size.width = template_size.height =
        (template_size.width + template_size.height) / 2.0f;
    rescale_ratio = sqrt(pow(params.template_size,2) / (template_size.width * template_size.height));
    if(rescale_ratio > 1)  {
        rescale_ratio = 1;
    }
    rescaled_template_size = Size2i(cvFloor(template_size.width * rescale_ratio),
            cvFloor(template_size.height * rescale_ratio));
    object_center = Point2f(static_cast<float>(boundingBox.x) + original_target_size.width / 2.0f,
            static_cast<float>(boundingBox.y) + original_target_size.height / 2.0f);
std::cout << "init 4"  << std::endl;
    yf = gaussian_shaped_labels(params.gsl_sigma,
            rescaled_template_size.width / cell_size, rescaled_template_size.height / cell_size);
    if(params.window_function.compare("hann") == 0) {
        window = get_hann_win(Size(yf.cols,yf.rows));
    } else if(params.window_function.compare("cheb") == 0) {
        window = get_chebyshev_win(Size(yf.cols,yf.rows), params.cheb_attenuation);
    } else if(params.window_function.compare("kaiser") == 0) {
        window = get_kaiser_win(Size(yf.cols,yf.rows), params.kaiser_alpha);
    } else {
        CV_Error(Error::StsBadArg, "Not a valid window function");
    }
std::cout << "init 5"  << std::endl;
    Size2i scaled_obj_size = Size2i(cvFloor(original_target_size.width * rescale_ratio / cell_size),
            cvFloor(original_target_size.height * rescale_ratio / cell_size));
	std::cout << "pky: rescale_ratio = " << rescale_ratio  << std::endl;
	std::cout << "pky: original_target_size = " << original_target_size.width << "*" << original_target_size.height << std::endl;
	std::cout << "pky: scaled_obj_size = " << scaled_obj_size.width << "*" << scaled_obj_size.height << std::endl;
	std::cout << "pky: rescaled_template_size = " << rescaled_template_size.width <<"*" << rescaled_template_size.height<< std::endl;
	std::cout << "pky: cell_size = " << cell_size << std::endl;
	//set dummy mask and area;
    int x0 = std::max((yf.size().width - scaled_obj_size.width)/2 - 1, 0);
    int y0 = std::max((yf.size().height - scaled_obj_size.height)/2 - 1, 0);
    default_mask = Mat::zeros(yf.size(), CV_32FC1);
    default_mask(Rect(x0,y0,scaled_obj_size.width, scaled_obj_size.height)) = 1.0f;
	std::cout << "pky: scaled_obj_size = " << scaled_obj_size << std::endl;
    default_mask_area = static_cast<float>(sum(default_mask)[0]);

    //initalize segmentation
    if(params.use_segmentation) 
	{
        //Mat hsv_img = bgr2hsv(image);
        //hist_foreground = Histogram(hsv_img.channels(), params.histogram_bins);
        //hist_background = Histogram(hsv_img.channels(), params.histogram_bins);
        //extract_histograms(hsv_img, bounding_box, hist_foreground, hist_background);
        //filter_mask = segment_region(hsv_img, object_center, template_size,
        //        original_target_size, current_scale_factor);

		hist_foreground = Histogram(image.channels(), params.histogram_bins);
		hist_background = Histogram(image.channels(), params.histogram_bins);
		extract_histograms(image, bounding_box, hist_foreground, hist_background);
		filter_mask = segment_region(image, object_center, template_size,
			original_target_size, current_scale_factor);

        //update calculated mask with preset mask
        if(preset_mask.data){
            Mat preset_mask_padded = Mat::zeros(filter_mask.size(), filter_mask.type());
            int sx = std::max((int)cvFloor(preset_mask_padded.cols / 2.0f - preset_mask.cols / 2.0f) - 1, 0);
            int sy = std::max((int)cvFloor(preset_mask_padded.rows / 2.0f - preset_mask.rows / 2.0f) - 1, 0);
            preset_mask.copyTo(preset_mask_padded(
                        Rect(sx, sy, preset_mask.cols, preset_mask.rows)));
            filter_mask = filter_mask.mul(preset_mask_padded);
        }
        erode_element = getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(1,1));

        resize(filter_mask, filter_mask, yf.size(), 0, 0, INTER_NEAREST);
        if(check_mask_area(filter_mask, default_mask_area)) {
            dilate(filter_mask , filter_mask, erode_element);
        } else {
            filter_mask = default_mask;
        }

    } else {
        filter_mask = default_mask;
    }

    //initialize filter
    Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
        cvFloor(current_scale_factor * template_size.height));
	//std::cout << "pky: chid u_resize1 = " << patch.size() << "  *  " << patch.channels() << std::endl;
	int64 t1 = cv::getTickCount();
    resize(patch, patch, rescaled_template_size, 0, 0, INTER_CUBIC);
	int64 t2 = cv::getTickCount();
	double tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	//std::cout << "pky: time_resize1 = " << tick_counter << std::endl;
	std::cout << "pky: chidu_resize1 = " << patch.size() << "  *  " << patch.channels() << std::endl;
	std::cout << "pky: rescaled_template_size = " << rescaled_template_size << std::endl;
    std::vector<Mat> patch_ftrs = get_features(patch, yf.size());
    std::vector<Mat> Fftrs = fourier_transform_features(patch_ftrs);
	t1 = cv::getTickCount();
    xfd_filter = create_xfd_filter(Fftrs, yf, filter_mask);
	t2 = cv::getTickCount();
	tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	std::cout << "pky: time_create_xfd_filter1 = " << tick_counter << std::endl;
	std::cout << "pky: create_xfd_filter_Fftrs_imagefeature.size() = " << Fftrs.size() << " yf = " << yf.size()<< "filter_mask = " << filter_mask.size() << std::endl;

    if(params.use_channel_weights) {
        Mat current_resp;
        filter_weights = std::vector<float>(xfd_filter.size());
        float chw_sum = 0;
        for (size_t i = 0; i < xfd_filter.size(); ++i) {
            mulSpectrums(Fftrs[i], xfd_filter[i], current_resp, 0, true);
            idft(current_resp, current_resp, DFT_SCALE | DFT_REAL_OUTPUT);
            double max_val;
            minMaxLoc(current_resp, NULL, &max_val, NULL , NULL);
            chw_sum += static_cast<float>(max_val);
            filter_weights[i] = static_cast<float>(max_val);
        }
        for (size_t i = 0; i < filter_weights.size(); ++i) {
            filter_weights[i] /= chw_sum;
        }
    }

	t1 = cv::getTickCount();
    //initialize scale search
    dsst = DSST(image, bounding_box, template_size, params.number_of_scales, params.scale_step,
            params.scale_model_max_area, params.scale_sigma_factor, params.scale_lr);
	t2 = cv::getTickCount();
	tick_counter = (t2 - t1) * 1000.0 / cv::getTickFrequency();
	std::cout << "pky: time_dsst1 = " << tick_counter << std::endl;

    model=makePtr<TrackerXFDModel>();
}

void TrackerXFDImpl::setOutThresholdParm(float th_psr, float th_scr)
{
	params.psr_threshold = th_psr;
	params.score_threshold = th_scr;
}

bool TrackerXFDImpl::getTrackerStatue()
{
	return flag_loss_target;
}

}  // namespace impl

TrackerXFD::Params::Params()
{
	use_channel_weights = false;
	use_segmentation = false;
	use_hog = false;
	use_cnn = false;
	use_color_names = false;
	use_gray = true;
	use_rgb = false;
	window_function = "hann";
	kaiser_alpha = 3.75f;
	cheb_attenuation = 45;
	padding = 5.0f;
	template_size = 200;//200;
	gsl_sigma = 1.0f;
	hog_orientations = 9;
	hog_clip = 0.2f;
	num_hog_channels_used = 18;//18
	num_cnn_channels_used = 64 + 128;
	filter_lr = 0.02f;//0.02f;
	weights_lr = 0.02f;
	admm_iterations = 4;
	number_of_scales = 33;//33
	scale_sigma_factor = 0.250f;
	scale_model_max_area = 512.0f;// 512.0f;
	scale_lr = 0.025f;//0.025f
	scale_step = 1.080f;//1.020f;
	histogram_bins = 16;
	background_ratio = 2;
	histogram_lr = 0.04f;
	psr_threshold = 0.0f;//0.035f 单次响应阈值
	score_threshold = 10.0f;//1.0f	响应质量得分 分数越高限制越小

}

TrackerXFD::TrackerXFD()
{
    // nothing
}

TrackerXFD::~TrackerXFD()
{
    // nothing
}

Ptr<TrackerXFD> TrackerXFD::create(const TrackerXFD::Params &parameters)
{
    return makePtr<TrackerXFDImpl>(parameters);
}

}}  // namespace
