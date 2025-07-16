/************************************************************************
 *FileName:  Midware_GPIO.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 中间件-GPIO方法
 ************************************************************************/
#ifndef __MIDWARE_GPIO_H__
#define __MIDWARE_GPIO_H__

#include "project_common.h"

#define SYSFS_GPIO_EXPORT              "/sys/class/gpio/export"
#define SYSFS_GPIO_UNEXPORT           "/sys/class/gpio/unexport"

int gpio_init(unsigned int gpio_chip_num, unsigned int gpio_offset_num, int direction);//0:in 1:out

int gpio_out_value(unsigned int gpio_chip_num, unsigned int gpio_offset_num,int value);

int gpio_in_value(unsigned int gpio_chip_num, unsigned int gpio_offset_num);

int gpio_set_edge(unsigned int gpio_chip_num, unsigned int gpio_offset_num,unsigned int edge);

#endif // __MIDWARE_GPIO_H__
