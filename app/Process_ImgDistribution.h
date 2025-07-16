/************************************************************************
 *FileName:  Process_ImgDistribution.h
 *Author:    Jin Qiuyu
 *Version:   2.1
 *Date:      2021-11-18
 *Description: 应用层-遥测帧接收类
 ************************************************************************/
#ifndef _PROCESS_IMGDISTRIBUTION_H
#define _PROCESS_IMGDISTRIBUTION_H

#include "project_common.h"
#include "Process_All_Base.h"
#include "Process_global.h"

class CProcess_ImgDistribution : public CProcess_All_Base
{
public:

	CProcess_ImgDistribution(void);
	~CProcess_ImgDistribution(void);

	CProcess_ImgDistribution& operator=(const CProcess_ImgDistribution&); // 禁止copy

	virtual unsigned long long Deal(void);

private:
	TimeTick *p_tick;
	int count_tick;

};

#endif //_PROCESS_IMGDISTRIBUTION_H