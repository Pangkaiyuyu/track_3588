/************************************************************************
 *FileName:  Midware_GPIO.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 中间件-GPIO方法
 ************************************************************************/
#ifndef __MIDWARE_SUPPLEMENT_H__
#define __MIDWARE_SUPPLEMENT_H__

#include "project_common.h"

class TimeTick
{
public:
	TimeTick(void);
	~TimeTick(void);
	unsigned long long GetTimeBase(void);
	unsigned long long GetTimeDiff(int Type = 0);
	unsigned long long TimeReset(void);
private:

	struct timeval m_old_time;
	struct timeval m_now_time;
	struct timeval m_start_time;
	struct timeval m_end_time;

};

void delay_ms(int time);

void uint8_setbit(unsigned char *input_uint8, unsigned char pos, char bit);

//----------------------------------------管理文件函数-------------------------------------------------------

// int fileNameFilter(const struct dirent *cur);
// 
// int files_input_sort(const char* path, int max_num, deque<char*> &files);
// 
// int files_log_init(const char* path, deque<char*> &files);
// 
// int files_input(const char* path, int max_num, deque<char*> &files);
// 
// void files_init();
// 
// void Sfiles_free(deque<char*> &files);
// 
// void files_free();
// 
// int gcc_get_Driver_dir(const char* path, char **files);
// 
// int gcc_TimeFindMediaFile(const char *StartTime, const char *EndTime, const char* path, deque<char *> &select_files, int *select_num);
// 
// void stampTime2strTime(long long stampTime, char* strTime);
// 
// long long strTime2stampTime(long long lstart_time);
// 
// int remove_dir(const char *dir);
// 
// int readFile(const char* fileName, unsigned char** dataPtr, int *len);


#endif // __MIDWARE_SUPPLEMENT_H__
