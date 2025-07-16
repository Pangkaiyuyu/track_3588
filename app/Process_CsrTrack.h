/************************************************************************
 *FileName:  Process_CsrTrack.h
 *Author:    Pang Kaiyu
 *Version:   1.0
 *Date:      2024-09-21
 *Description: 加深层特征的CSR跟踪放到PCIE接收线程上
 ************************************************************************/
#ifndef _PROCESS_CSRTRACK_H
#define _PROCESS_CSRTRACK_H

#include "project_common.h"
#include "Process_All_Base.h"
#include "Process_global.h"

class CProcess_CsrTrack : public CProcess_All_Base
{
public:

	CProcess_CsrTrack();
	~CProcess_CsrTrack();

	unsigned long long Deal();
	 

};

#endif //_PROCESS_CSRTRACK_H