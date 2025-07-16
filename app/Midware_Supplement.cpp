/************************************************************************
 *FileName:  Midware_Supplement.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 中间件-辅助方法
 ************************************************************************/
#include "Midware_Supplement.h"

TimeTick::TimeTick(void)
{
	memset(&m_start_time, 0, sizeof(struct timeval));
	memset(&m_end_time, 0, sizeof(struct timeval));
	memset(&m_now_time, 0, sizeof(struct timeval));
	memset(&m_old_time, 0, sizeof(struct timeval));
}

TimeTick::~TimeTick(void)
{

}

unsigned long long TimeTick::GetTimeBase(void)
{
	unsigned long long m_time_now = 0;

	gettimeofday(&m_start_time, NULL);

	m_time_now = m_start_time.tv_sec * 1000 + m_start_time.tv_usec / 1000.0;

	return m_time_now;
}

unsigned long long TimeTick::GetTimeDiff(int Type)
{
	unsigned long long m_during_time = 0;

	m_old_time.tv_sec = m_now_time.tv_sec;
	m_old_time.tv_usec = m_now_time.tv_usec;

	if (Type == 0)
	{
		gettimeofday(&m_end_time, NULL);
		m_during_time = (m_end_time.tv_sec - m_start_time.tv_sec) * 1000 + (m_end_time.tv_usec - m_start_time.tv_usec) / 1000.0;
		return m_during_time;
	}
	else if (Type == 1)
	{
		gettimeofday(&m_now_time, NULL);
		m_during_time = (m_now_time.tv_sec - m_old_time.tv_sec) * 1000 + (m_now_time.tv_usec - m_old_time.tv_usec) / 1000.0;
		return m_during_time;
	}
	else
	{
		return 0;
	}
}

unsigned long long TimeTick::TimeReset(void)
{
	memset(&m_start_time, 0, sizeof(struct timeval));
	memset(&m_end_time, 0, sizeof(struct timeval));
	memset(&m_now_time, 0, sizeof(struct timeval));
	memset(&m_old_time, 0, sizeof(struct timeval));

	return 0;
}

void delay_ms(int time)
{
	struct timeval tv;

	long int begin, stop;

	gettimeofday(&tv, NULL);
	begin = tv.tv_usec;

	//printf("start time = %ld\r\n", start);

	do
	{
		gettimeofday(&tv, NULL);
		stop = tv.tv_usec;

		if (stop < begin)
		{
			stop = stop + 1000000;
		}

		//printf("stop time = %ld\r\n", stop);
	} while ((stop - begin) < time * 1000);
}


void uint8_setbit(unsigned char *input_uint8, unsigned char pos, char bit)
{
	if (input_uint8 == NULL)
	{
		return;
	}

	unsigned int temp = 0;

	switch (pos)
	{
		case 0:
		{
			temp = 0x01;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;
		}
		case 1:
		{
			temp = 0x02;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;

		}
		case 2:
		{
			temp = 0x04;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;
		}
		case 3:
		{
			temp = 0x08;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;
		}
		case 4:
		{
			temp = 0x10;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;
		}
		case 5:
		{
			temp = 0x20;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;
		}
		case 6:
		{
			temp = 0x40;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;
		}
		case 7:
		{
			temp = 0x80;

			if (bit) {
				*input_uint8 = *input_uint8 | temp;
			}
			else {
				*input_uint8 = *input_uint8 & ~temp;
			}
			break;
		}
		default:
		{
			break;
		}
	}
}