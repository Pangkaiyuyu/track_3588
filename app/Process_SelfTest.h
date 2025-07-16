/************************************************************************
 *FileName:  Process_SelfTest.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 应用层-自测试使用
 ************************************************************************/
#ifndef _CPROCESS_SELFTEST_H
#define _CPROCESS_SELFTEST_H

#include "project_common.h"
#include "Process_All_Base.h"
#include "Process_global.h"

class CProcess_SelfTest : public CProcess_All_Base
{
public:

	CProcess_SelfTest(void);
	~CProcess_SelfTest(void);
	CProcess_SelfTest& operator=(const CProcess_SelfTest&); // 禁止copy

	virtual unsigned long long Deal(void);

private:

	struct termios initial_settings, new_settings;
	int peek_character;
	void init_keyboard();
	void close_keyboard();
	int kbhit();
	int readch();
};

#endif //_CPROCESS_SELFTEST_H