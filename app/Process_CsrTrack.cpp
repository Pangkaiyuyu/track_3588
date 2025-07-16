/************************************************************************
 *FileName:  Process_CsrTrack.cpp
 *Author:    Pang Kaiyu
 *Version:   1.0
 *Date:      2024-09-21
 *Description: 加深层特征的CSR跟踪放到PCIE接收线程上
 ************************************************************************/
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <fstream>
#include <unistd.h>
using namespace std;
using namespace cv;

#include "Process_CsrTrack.h"
#include "Algorithm_Tracker.h"

CProcess_CsrTrack::CProcess_CsrTrack()
{

}

CProcess_CsrTrack::~CProcess_CsrTrack()
{

}

unsigned long long CProcess_CsrTrack::Deal()
{
	LOG(INFO) << "CProcess_CsrTrack start";

    int n_start_frame = 1;
	int64 t1 = 0;
	int64 t2 = 0;
	int64 tick_counter=0;
    Mat ImgSrc;
    cv::Size m_720p_size(1280, 720);

	Ptr<TrackerXFD> tracker = TrackerXFD::create();
	Rect roi(0, 0, 0, 0);
	TimeTick *m_tick = new TimeTick();
	img_pack * p_temp_720;

	while (flag_end != 1)
	{
		
		p_temp_720 = queue_img_src_to_tracker.ReadDat();

		if (p_temp_720 == NULL)
		{
			usleep(10);
			continue;
		}
		else
		{
			printf("the frame  is %d\n", n_start_frame);
			Mat frame(m_720p_size, CV_8UC3, p_temp_720->p_buf);
			printf("111\n");
			if (b_track_init == true)
			{

				int m_track_xsc = (int)(440);
				int m_track_ysc = (int)(439);
				int m_track_wsc = (int)(99);
				int m_track_hsc = (int)(169);

				LOG(INFO) << "m_track_xsc = " << m_track_xsc;
				LOG(INFO) << "m_track_ysc = " << m_track_ysc;
				LOG(INFO) << "m_track_wsc = " << m_track_wsc;
				LOG(INFO) << "m_track_hsc = " << m_track_hsc;

				roi = cv::Rect(m_track_xsc, m_track_ysc, m_track_wsc, m_track_hsc);
				m_tick->GetTimeBase();

				tracker->init(frame, roi);

				LOG(INFO) << "init :"<<m_tick->GetTimeDiff(0);
				b_track_init = false;
				roi_1 = roi.x;       // 左上角的 x 坐标
				roi_2 = roi.y;       // 左上角的 y 坐标
				roi_3 = roi.width;   // 矩形的宽度
				roi_4 = roi.height; 

			}
			else
			{
				m_tick->GetTimeBase();

				
				if (tracker->getTrackerStatue() == true)
				{
					LOG(INFO) << "flag_loss_target !!!!!!";

					b_track_init = false;
					delete p_temp_720;
					p_temp_720 = NULL;

					frame.release();
					continue;
				}

				tracker->update(frame, roi);

				roi_1 = roi.x;       // 左上角的 x 坐标
				roi_2 = roi.y;       // 左上角的 y 坐标
				roi_3 = roi.width;   // 矩形的宽度
				roi_4 = roi.height; 

				LOG(INFO) << "update :" << m_tick->GetTimeDiff(0);
			}

			delete p_temp_720;
			p_temp_720 = NULL;
			frame.release();
			n_start_frame++;
		}


	}

	//delete tracker;

	return 0;
}
