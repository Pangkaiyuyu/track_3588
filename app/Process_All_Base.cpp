/************************************************************************
 *FileName:  Process_All_Base.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 应用层-构建基类
 ************************************************************************/

#include "Process_All_Base.h"

CProcess_All_Base::CProcess_All_Base(void)
{
	nPriority = 50;
	b_CPU_chs = false;
	nCpu_core = 0;
	m_pThread = 0;
	m_inher = 0;

	memset(&m_attr, 0, sizeof(pthread_attr_t));
	memset(&m_param, 0, sizeof(struct sched_param));

	nCpu_core_ALL = 0;
	InterIO_deal = NULL;

	nCpu_core_ALL = sysconf(_SC_NPROCESSORS_CONF);
}

CProcess_All_Base::~CProcess_All_Base(void)
{

}

void CProcess_All_Base::Start_Thread(void)
{
	InterIO_deal = new DEALIO();

	InterIO_deal->b_CPU_chs = b_CPU_chs;

	if (nCpu_core < nCpu_core_ALL)
	{
		InterIO_deal->nCpu_core = nCpu_core;
	}
	else
	{
		InterIO_deal->nCpu_core = 0;
	}
		

	InterIO_deal->Addr = this;

	//初始化线程属性
	pthread_attr_init(&m_attr);
	//获取继承的调度策略
	pthread_attr_getinheritsched(&m_attr, &m_inher);

	//必需设置inher的属性为 PTHREAD_EXPLICIT_SCHED,否则设置线程的优先级会被忽略

	if (m_inher == PTHREAD_EXPLICIT_SCHED)
	{
	}
	else if (m_inher == PTHREAD_INHERIT_SCHED)
	{
		m_inher = PTHREAD_EXPLICIT_SCHED;
	}
	else
	{

	}

	pthread_attr_setinheritsched(&m_attr, m_inher);
	//设置线程调度策略
	pthread_attr_setschedpolicy(&m_attr, SCHED_FIFO);
	//设置调度参数
	m_param.sched_priority = nPriority;
	pthread_attr_setschedparam(&m_attr, &m_param);

	pthread_create(&m_pThread, &m_attr,Thread_Function,InterIO_deal);
	//pthread_create(&m_pThread, NULL, Thread_Function, InterIO_deal);

}

void CProcess_All_Base::Wait_Thread_Exit(void)
{
	if (m_pThread != 0)
	{
		pthread_join(m_pThread, NULL);
		pthread_attr_destroy(&m_attr);
	}

	delete InterIO_deal;
	InterIO_deal = NULL;

}

void * CProcess_All_Base::Thread_Function(void* pParam)// 实现：指定核 + 调用Deal
{
	DEALIO *InterIO_temp = (DEALIO *)pParam;

	if (InterIO_temp->b_CPU_chs == true)
	{
		cpu_set_t m_mask;
		CPU_ZERO(&m_mask);
		CPU_SET(InterIO_temp->nCpu_core, &m_mask);

		if (pthread_setaffinity_np(pthread_self(), sizeof(m_mask), &m_mask) < 0)
		{
			//printf("set thread affinity failed\n");
		}
	}

	return (void *)(((CProcess_All_Base*)InterIO_temp->Addr)->Deal());
}

void CProcess_All_Base::Config_Thread(int Priority, bool IsCPU_Chs, int CPU_core)
{
	nPriority = Priority;
	b_CPU_chs = IsCPU_Chs;
	nCpu_core = CPU_core;
}
