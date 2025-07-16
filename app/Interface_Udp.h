/************************************************************************
 *FileName:  Interface_Udp.h
 *Author:    Jin Qiuyu
 *Version:   1.0
 *Date:      2017-9-27
 *Description: 接口子类
 ************************************************************************/

#ifndef _INTERFACE_UDP_H
#define _INTERFACE_UDP_H

#include "project_common.h"

//540 1740 65507
#define ONE_PACKAGE_LENGTH 65507

class CUdp
{
public:
	CUdp(const char *Local_Addr			= "127.0.0.1"		,	//本地IP
		 unsigned int  Local_Port		= 6666				,	//本地端口号
		 const char * Host_Addr			= "192.168.100.1"	,	//发送端的IP
		 unsigned int Host_Port			= 8888				,	//发送端的端口号
		 unsigned int BufSizeW			= 0					,	//写缓冲区大小
		 unsigned int BufSizeR			= 0					,	//读缓冲区大小
		 bool BlockMode					= true				,	//阻塞模式
		 int Overtime					= -1					//阻塞时长
	);

	~CUdp(void);


	int OpenDev(void *ParaInfo);
	int Config(void *ParaInfo);	
	void CloseDev(void *ParaInfo);
	int ResetDev(void *ParaInfo);
	int WriteDat(void *pBuf, int Length, void *OtherInfo);
	int ReadDat(void *pBuf, int Length, void *OtherInfo);//读取长度不能超过65507，根据需求自行拼帧

	
public:

	int Init(void);

private:

	char n_LocalAddr[32];
	int n_LocalPort;

	char n_HostAddr[32];
	int n_HostPort;

	int n_BufSizeW;
	int n_BufSizeR;

	bool n_BlockMode;

	int n_Overtime;

	int udp_fd;

	struct sockaddr_in Udp_LocalAddr;

	struct sockaddr_in Udp_HostAddr;
	int HostAddr_lens;

	struct sockaddr_in Udp_RcvAddr;
	socklen_t RevAddr_lens;

	int Udp_Init_Finished;

};

#endif // _INTERFACE_UDP_H
