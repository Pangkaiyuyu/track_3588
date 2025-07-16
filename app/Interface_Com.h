/************************************************************************
 *FileName:  Interface_Com.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 接口子类-串口
 ************************************************************************/
#ifndef _INTERFACE_COM_H
#define _INTERFACE_COM_H

#include "project_common.h"

class CCom
{
public:
	CCom(int ComId = 0,	// 串口号
		unsigned long dwBaudRate = 9600,// 波特率
		char Parity = 'O',	// 校验位
		int ByteSize = 8,	// 数据位
		int StopBits = 1,	// 停止位
		bool Block = true, //阻塞模式
		int Overtime = -1); //阻塞时长;

	~CCom(void);

	int OpenDev(void *ParaInfo);
	int Config(void *ParaInfo);	
	int CloseDev(void *ParaInfo);

	int WriteDat(void *pBuf, int Length, void *OtherInfo);
	int ReadDat(void *pBuf, int Length, void *OtherInfo);

	int ResetDev(void *ParaInfo);

	int Init(void);//初始化：打开并配置串口

private:

	int m_ComId;	// 串口号
	unsigned long m_dwBaudRate;	// 波特率
	char m_Parity;	// 校验位
	int m_ByteSize;	// 数据位
	int m_StopBits;	// 停止位
	bool m_Block; //阻塞模式

	int m_Overtime;
	int com_fd;

	int Com_Init_Finished;//初始化完毕

};

#endif // _INTERFACE_COM_H