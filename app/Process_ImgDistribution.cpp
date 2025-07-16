/************************************************************************
 *FileName:  Process_ImgDistribution.cpp
 *Author:    Jin Qiuyu
 *Version:   2.1
 *Date:      2021-11-18
 *Description: 应用层-遥测帧接收类
 ************************************************************************/

#include "Process_ImgDistribution.h"

//#define  TEST_DATA 1

CProcess_ImgDistribution::CProcess_ImgDistribution(void)
{
	p_tick = NULL;
	count_tick = 0;
}

CProcess_ImgDistribution::~CProcess_ImgDistribution(void)
{

}

unsigned long long CProcess_ImgDistribution::Deal(void)
{
	LOG(INFO) << "CProcess_ImgDistribution start";

	unsigned int n_pic_lens_ch = 1280 * 720 * 3; //1920 * 1080 * 3;
	unsigned int yuv_Lens = 1280 * 720 * 3;
	p_tick = new TimeTick();

	unsigned char test_pix = 18;
	unsigned int test_frame_count = 0;

	Mat yuvImg;
	int count_frame = 0;
	
	while (flag_end != 1)
	{

#if TEST_DATA

		img_pack *p_temp = new img_pack(n_pic_lens_ch);
		memset(p_temp->p_buf, test_pix, n_pic_lens_ch);
		test_pix++;
		if (test_pix >= 200)
		{
			test_pix = 18;
		}

		//memcpy(p_temp->p_buf, dst.data, n_pic_lens_ch);

		p_temp->n_length = n_pic_lens_ch;
		queue_img_src_to_vpss.WriteDat(p_temp);

		usleep(30 * 1000);
		LOG(INFO) <<  "img_read : frame_frq : "<<p_tick->GetTimeDiff(1);
		test_frame_count++;
#else
		n_pic_lens_ch = 1280 * 720 * 3;
		//n_pic_lens_ch = 1024 * 1024;

		img_pack *p_temp = new img_pack(n_pic_lens_ch);
		int ret = m_pcie_p2p->ReadDat(p_temp->p_buf, n_pic_lens_ch, NULL);
		p_temp->n_length = n_pic_lens_ch;

		if (ret == -1)
		{
			LOG(INFO) << "pcie read failed!";
			delete p_temp;
			p_temp = NULL;
			continue;
		}

		
		// FILE* fp = fopen("test.bin", "wb");
		// fwrite(p_temp->p_buf, 1, n_pic_lens_ch, fp);
		// fclose(fp);
		//delete p_temp;
		//p_temp = NULL;

		LOG(INFO) << "img_read : frame_frq : " << p_tick->GetTimeDiff(1);
		test_frame_count++;
		count_frame++;

		img_pack *p_temp2 = new img_pack(n_pic_lens_ch);
		memcpy(p_temp2->p_buf,p_temp->p_buf,n_pic_lens_ch);
		p_temp2->n_length = n_pic_lens_ch;


		queue_img_src_to_vpss.WriteDat(p_temp);
		queue_img_src_to_tracker.WriteDat(p_temp2);

		LOG(INFO) << "count_frame : " << count_frame;
		
#endif

	}

	if (p_tick != NULL)
	{
		delete p_tick;
		p_tick = NULL;
	}

	return 0;
}