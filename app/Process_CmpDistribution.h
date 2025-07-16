/************************************************************************
 *FileName:  Process_CmpPostback.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2021-08-08
 *Description: 应用层-压缩打包分发类
 ************************************************************************/
#ifndef _PROCESS_CMPDISTRIBUTION_H
#define _PROCESS_CMPDISTRIBUTION_H

#include "project_common.h"
#include "Process_All_Base.h"
#include "Process_global.h"

class CProcess_CmpDistribution : public CProcess_All_Base
{
public:

	CProcess_CmpDistribution(void);
	~CProcess_CmpDistribution(void);
	CProcess_CmpDistribution& operator=(const CProcess_CmpDistribution&); // 禁止copy

	virtual unsigned long long Deal(void);

private:

	int n_frame_count;
	TimeTick *p_tick;
};

#endif //_PROCESS_CMPDISTRIBUTION_H