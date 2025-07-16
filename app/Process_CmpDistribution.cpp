/************************************************************************
 *FileName:  Process_CmpPostback.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2021-08-08
 *Description: 应用层-压缩打包回传类
 ************************************************************************/

#include "Process_CmpDistribution.h"

CProcess_CmpDistribution::CProcess_CmpDistribution(void)
{
	n_frame_count = 0;
	p_tick = NULL;
}

CProcess_CmpDistribution::~CProcess_CmpDistribution(void)
{
	
}

unsigned long long CProcess_CmpDistribution::Deal(void)
{
	LOG(INFO) << "CProcess_CmpDistribution start";
	stream_pack *temp = NULL;
	int ret = 0;
	int n_pack_num = 0;
	int n_left_lenth = 0;
	cmp_send_protocol *p_cmp_send = new cmp_send_protocol;
	memset(p_cmp_send, 0, sizeof(cmp_send_protocol));
	p_cmp_send->cmp_head_1 = 0xEE;
	p_cmp_send->cmp_head_2 = 0x16;

	unsigned char *p_pcie_send = new unsigned char[MAX_CMP_PACK_LENGTH];
	p_tick = new TimeTick();

	while (flag_end != 1)
	{
		temp = queue_stream_to_out.ReadDat();

		if (temp != NULL)
		{
			n_pack_num = (int)ceil(temp->n_length / CMP_PACK_LENGTH_F);//保护前几帧过大的情况
			if (n_pack_num > 64)
			{
				LOG(WARNING) << "FRAME TOO LARGE!!!!!!! : " << n_pack_num * 61440;
				delete temp;
				temp = NULL;
				continue;
			}

			memset(p_pcie_send, 0, MAX_CMP_PACK_LENGTH);
			p_cmp_send->cmp_pack_num = n_pack_num;
			n_frame_count++;
			n_left_lenth = temp->n_length;

			for (int n_pack_count = 0; n_pack_count < n_pack_num; n_pack_count++)
			{
				if (n_pack_count == n_pack_num - 1)//最后一包
				{
					memset(p_cmp_send->cmp_data, 0x00, CMP_PACK_LENGTH);
					memcpy(p_cmp_send->cmp_data, temp->p_buf + n_pack_count * CMP_PACK_LENGTH, n_left_lenth);
					p_cmp_send->cmp_pack_count = n_pack_count;
					p_cmp_send->cmp_length_now = n_left_lenth;
					memcpy(p_pcie_send + n_pack_count * 61440, p_cmp_send, 61440);
					m_udp_p2p->WriteDat(p_cmp_send, sizeof(cmp_send_protocol), NULL);
				}
				else
				{
					memcpy(p_cmp_send->cmp_data, temp->p_buf + n_pack_count * CMP_PACK_LENGTH, CMP_PACK_LENGTH);
					p_cmp_send->cmp_pack_count = n_pack_count;
					p_cmp_send->cmp_length_now = CMP_PACK_LENGTH;
					n_left_lenth = n_left_lenth - CMP_PACK_LENGTH;
					memcpy(p_pcie_send + n_pack_count * 61440, p_cmp_send, 61440);
					m_udp_p2p->WriteDat(p_cmp_send, sizeof(cmp_send_protocol), NULL);
				}
			}

			delete temp;
			temp = NULL;
		}
		else
		{
			usleep(1000);
		}
		

	}

	delete p_cmp_send;
	delete []p_pcie_send;

	return 0;
}
