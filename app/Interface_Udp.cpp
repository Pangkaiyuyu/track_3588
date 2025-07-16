/************************************************************************
 *FileName:  Interface_Udp.h
 *Author:    Jin Qiuyu
 *Version:   1.0
 *Date:      2017-9-27
 *Description: 接口子类
 ************************************************************************/

#include "Interface_Udp.h"

CUdp::CUdp(const char *Local_Addr, unsigned int  Local_Port, const char * Host_Addr, unsigned int Host_Port,
	unsigned int BufSizeW, unsigned int BufSizeR, bool BlockMode, int Overtime)
{
	memset(n_LocalAddr, 0, sizeof(n_LocalAddr));

	if (Local_Addr != NULL)
	{
		strncpy(n_LocalAddr, Local_Addr, 20);
	}

	n_LocalPort = Local_Port;

	memset(n_HostAddr, 0, sizeof(n_HostAddr));

	if (Host_Addr != NULL)
	{
		strncpy(n_HostAddr, Host_Addr, 20);
	}


	n_HostPort = Host_Port;
	n_BufSizeW = BufSizeW;
	n_BufSizeR = BufSizeR;
	n_BlockMode = BlockMode;
	n_Overtime = Overtime;

	udp_fd = 0;

	bzero(&Udp_LocalAddr, sizeof(Udp_LocalAddr));
	bzero(&Udp_HostAddr, sizeof(Udp_HostAddr));
	HostAddr_lens = 0;

	bzero(&Udp_RcvAddr, sizeof(Udp_RcvAddr));

	RevAddr_lens = 0;
	Udp_Init_Finished = 0;

}


CUdp::~CUdp(void)
{

}

int CUdp::Init(void)
{
	if(Udp_Init_Finished == 0)
	{
		if( OpenDev(NULL) != 0)
		{
			printf("Fail to Open Udp!\n");
			return -1;
		}

		if( Config(NULL) != 0)
		{
			printf("Fail to Config Udp!\n");
			return -2;
		}
		Udp_Init_Finished = 1;
	}
	else
	{
		printf("Init have been already done!\n");
	}
	
	return 0;
}

int CUdp::OpenDev(void *ParaInfo)
{
	udp_fd = socket(AF_INET,SOCK_DGRAM,0);//IPV4 + UDP

	if(udp_fd < 0)
	{
		printf("create socket fail!\n");
		return -1;
	}

	//if(n_BlockMode == false) //非阻塞模式
	//{
	//	if(fcntl(udp_fd,F_SETFL,FNDELAY) < 0)
	//	{
	//		printf("fcntl(unblock) failed!\n");
	//		return -2;
	//	}
	//}
	//else
	//{
	//	if(fcntl(udp_fd,F_SETFL,0) < 0)
	//	{
	//		printf("fcntl(block) failed!\n");
	//		return -2;
	//	}	
	//}

	return 0;
}

int CUdp::Config(void *ParaInfo)
{
	//服务端IP信息
	memset(&Udp_HostAddr, 0, sizeof(Udp_HostAddr));
	Udp_HostAddr.sin_family = AF_INET;
	Udp_HostAddr.sin_addr.s_addr = inet_addr(n_HostAddr);
	Udp_HostAddr.sin_port = htons(n_HostPort);

	//本地IP信息
	memset(&Udp_LocalAddr, 0, sizeof(Udp_LocalAddr));
	Udp_LocalAddr.sin_family = AF_INET;
	Udp_LocalAddr.sin_addr.s_addr = inet_addr(n_LocalAddr);
	Udp_LocalAddr.sin_port = htons(n_LocalPort);

	HostAddr_lens = sizeof(Udp_HostAddr);

	int ret = bind(udp_fd,(struct sockaddr*)&Udp_LocalAddr,sizeof(Udp_LocalAddr));
	
	if(ret < 0)
	{
		printf("Socket bind fail!\n");
		return -1;
	}

	//if( setsockopt(udp_fd,SOL_SOCKET,SO_SNDBUF,(const char*)&n_BufSizeW,sizeof(int)) < 0 )
	//{
	//	printf("fail to set SO_SNDBUF!\n");
	//	return -2;
	//}

	//if( setsockopt(udp_fd,SOL_SOCKET,SO_RCVBUF,(const char*)&n_BufSizeR,sizeof(int)) < 0 )
	//{
	//	printf("fail to set SO_RCVBUF!\n");
	//	return -2;
	//}

	return 0;
}

int CUdp::WriteDat(void *pBuf, int Length, void *OtherInfo)
{
	int Size_real = 0;
	int n_Left_Length = 0;
	int n_Package_Num = 0;
	int ret = 0;

	if (Length > ONE_PACKAGE_LENGTH)
	{
		n_Package_Num = Length / ONE_PACKAGE_LENGTH;
		n_Left_Length = Length % ONE_PACKAGE_LENGTH;

		for (int count_temp = 0; count_temp < n_Package_Num; count_temp++)
		{
			ret = sendto(udp_fd, (char*)pBuf + count_temp * ONE_PACKAGE_LENGTH, ONE_PACKAGE_LENGTH, 0, (struct sockaddr*)&Udp_HostAddr, HostAddr_lens);
			if (ret == -1)
			{
				return -1;
			}
			else
			{
				Size_real += ret;
			}
		}

		ret = sendto(udp_fd, (char*)pBuf + n_Package_Num * ONE_PACKAGE_LENGTH, n_Left_Length, 0, (struct sockaddr*)&Udp_HostAddr, HostAddr_lens);
		if (ret == -1)
		{
			return -1;
		}
		else
		{
			Size_real += ret;
		}
	}
	else
	{
		ret = sendto(udp_fd, (char*)pBuf, Length, 0, (struct sockaddr*)&Udp_HostAddr, HostAddr_lens);
		if (ret == -1)
		{
			return -1;
		}
		else
		{
			Size_real += ret;
		}
	}

	return Size_real;
}

int CUdp::ReadDat(void *pBuf, int Length, void *OtherInfo)
{
	int ret = -1;

	if (n_BlockMode == false)//不阻塞模式 
	{
		int nfds;
		struct timeval tv;
		int Done_Length = 0;

		if (n_Overtime > 0)//Overtime ms 的超时时间
		{
			tv.tv_sec = (int)(n_Overtime / 1000.0);
			tv.tv_usec = n_Overtime % 1000;
		}
		else
		{
			tv.tv_sec = 1;
			tv.tv_usec = 0;
		}

		fd_set read_fd_set;
		FD_ZERO(&read_fd_set);
		FD_SET(udp_fd, &read_fd_set);

		nfds = select(udp_fd + 1, &read_fd_set, NULL, NULL, &tv);

		if (nfds == 0)
		{
			return ret;
		}
		else
		{
			ret = recvfrom(udp_fd, (char *)pBuf, Length, 0, (struct sockaddr*)&Udp_RcvAddr, &RevAddr_lens);
		}
	}
	else//永久阻塞模式 
	{
		ret = recvfrom(udp_fd, (char *)pBuf, Length, 0, (struct sockaddr*)&Udp_RcvAddr, &RevAddr_lens);
	}

	return ret;
}

void CUdp::CloseDev(void *ParaInfo)
{
	if (Udp_Init_Finished)
	{
		close(udp_fd);
		udp_fd = 0;
		Udp_Init_Finished = 0;
	}
}

int CUdp::ResetDev(void *ParaInfo)
{
	return 0;
}