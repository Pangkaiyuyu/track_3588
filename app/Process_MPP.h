/************************************************************************
 *FileName:  Midware_GPIO.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 编码组件-MPP
 ************************************************************************/
#ifndef _CPROCESS_MPP_H
#define _CPROCESS_MPP_H

#include "project_common.h"
#include "Process_All_Base.h"
#include "Process_global.h"


int test_mpp(int argc, char **argv);//   ./m_u204_app.out -i ./dvpp_vpc_1920x1080_nv12.yuv -t 7 -n 250 -o ./Test.h264 -w 1920 -h 1080 -fps 25

int mpp_init();


#endif // _CPROCESS_MPP_H
