/************************************************************************
 *FileName:  Process_All_Base.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 应用层-构建基类
 ************************************************************************/
#ifndef _PROCESS_ALL_BASE_H
#define _PROCESS_ALL_BASE_H

#include "project_common.h"

class CProcess_All_Base
{

public:
	CProcess_All_Base(void);
	virtual ~CProcess_All_Base(void);
	CProcess_All_Base& operator=(const CProcess_All_Base&); // 禁止copy

	void Config_Thread(int Priority, bool IsCPU_Chs, int CPU_core);//配置线程参数

	void Start_Thread(void);//开线程

	void Wait_Thread_Exit(void);

private:

	static void * Thread_Function(void* pParam);// 实现：指定核 + 调用Deal

	virtual unsigned long long Deal(void) { return 0; }

	typedef struct _Deal_IO
	{
		bool b_CPU_chs;
		int nCpu_core;
		void * Addr;
	}DEALIO;


	int nPriority;//指定优先级

	bool b_CPU_chs; //是否选定核，如果不选定则自动默认
	int nCpu_core;//指定核
	 
	pthread_t m_pThread;

	//线程优先级
	int m_inher;
	pthread_attr_t m_attr;
	struct sched_param m_param;

	//线程绑定核
	int nCpu_core_ALL;
	DEALIO *InterIO_deal;
};


#endif // _PROCESS_ALL_BASE_H
