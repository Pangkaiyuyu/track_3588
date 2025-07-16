/************************************************************************
 *FileName:  Log_cplus.cpp
 *Author:    Jin Qiuyu
 *Description: 中间件-日志管理
 ************************************************************************/
#ifndef __MIDWARE_GLOG_H__
#define __MIDWARE_GLOG_H__

#include "project_common.h"

#define GLOG_USE_GLOG_EXPORT

#include <glog/logging.h>
#include <glog/log_severity.h>


void Init_Log(void);

void Quit_Log(void);

void printDebug(void);

#endif // __MIDWARE_GLOG_H__