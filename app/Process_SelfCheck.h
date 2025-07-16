/************************************************************************
 *FileName:  Process_SelfCheck.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 应用层-自检类
 ************************************************************************/
#ifndef _CPROCESS_SELFCHECK_H
#define _CPROCESS_SELFCHECK_H

#include "project_common.h"
#include "Process_All_Base.h"
#include "Process_global.h"

class CProcess_SelfCheck : public CProcess_All_Base
{
public:

	CProcess_SelfCheck(void);
	~CProcess_SelfCheck(void);
	CProcess_SelfCheck& operator=(const CProcess_SelfCheck&); // 禁止copy

	virtual unsigned long long Deal(void);

private:

	//获取第N项开始的指针
	const char* get_items(const char*buffer, unsigned int item);

	//获取总的CPU时间
	unsigned long get_cpu_total_occupy();

	//获取进程的CPU时间
	unsigned long get_cpu_proc_occupy(unsigned int pid);

	//获取CPU占用率
	float get_proc_cpu(unsigned int pid);

	//获取进程占用内存
	unsigned int get_proc_mem(unsigned int pid);

	//获取进程占用虚拟内存
	unsigned int get_proc_virtualmem(unsigned int pid);

	//进程本身
	int get_pid(const char* process_name);

};

int Sys_Init();
int Interface_Init();
int Interface_Quit();

#endif //_CPROCESS_SELFCHECK_H