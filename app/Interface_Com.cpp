/************************************************************************
 *FileName:  Interface_Com.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2021-03-08
 *Description: 接口子类-串口
 ************************************************************************/
#include "Interface_Com.h"

CCom::CCom(int ComId,	//串口号
	unsigned long dwBaudRate,	//波特率
	char Parity,	//校验位
	int ByteSize,	//数据位
	int StopBits,	//停止位
	bool Block, //阻塞模式
	int Overtime)//阻塞时长
{
	m_ComId = ComId;
	m_dwBaudRate = dwBaudRate;
	m_Parity = Parity;
	m_ByteSize = ByteSize;
	m_StopBits = StopBits;
	m_Block = Block;

	m_Overtime = Overtime;

	Com_Init_Finished = 0;
	com_fd = 0;

}

CCom::~CCom(void)
{

}

int CCom::Init(void)
{
	if(Com_Init_Finished == 0)
	{
		if( OpenDev(NULL) != 0)
		{
			printf("Fail to Open Dev!\n");
			return -1;
		}

		if( Config(NULL) != 0)
		{
			printf("Fail to Config Dev!\n");
			return -2;
		}
		Com_Init_Finished = 1;
	}
	else
	{
		printf("Init have been already done!\n");
	}

	return 0;
}

int CCom::OpenDev(void *ParaInfo)
{
	switch (m_ComId)
	{
		case 0:
		{
			com_fd = open("/dev/ttyAMA0", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 1:
		{
			com_fd = open("/dev/ttyAMA1", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 2:
		{
			com_fd = open("/dev/ttyAMA2", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 3:
		{
			com_fd = open("/dev/ttyAMA3", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 4:
		{
			com_fd = open("/dev/ttyAMA4", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 5:
		{
			com_fd = open("/dev/ttyAMA5", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 6:
		{
			com_fd = open("/dev/ttyAMA6", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 7:
		{
			com_fd = open("/dev/ttyAMA7", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		case 8:
		{
			com_fd = open("/dev/ttyAMA8", O_RDWR | O_NOCTTY | O_NDELAY);
			break;
		}
		default:
		{
			printf("Unsupport m_ComId = %d!\n", m_ComId);
			return -1;
		}

	}

	if (-1 == com_fd)
	{
		printf("Can't Open Serial Port %d !", m_ComId);
		return -2;
	}

	//if(m_Block == false) //非阻塞模式
	//{
	//	if(fcntl(com_fd,F_SETFL,FNDELAY) < 0)
	//	{
	//		printf("fcntl(unblock) failed!\n");
	//		return -3;
	//	}
	//}
	//else
	//{
	//	if(fcntl(com_fd,F_SETFL,0) < 0)
	//	{
	//		printf("fcntl(unblock) failed!\n");
	//		return -3;
	//	}	
	//}

	if (fcntl(com_fd, F_SETFL, FNDELAY) < 0)
	{
		printf("fcntl(unblock) failed!\n");
		return -3;
	}
	return 0;
}

int CCom::Config(void *ParaInfo)
{
	struct termios newtio, oldtio;

	if (tcgetattr(com_fd, &oldtio) != 0)
	{
		perror("SetupSerial 1\n");
		return -1;
	}

	newtio = oldtio;
	cfmakeraw(&newtio);

	newtio.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
	newtio.c_oflag &= ~OPOST;
	newtio.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
	newtio.c_cflag &= ~(CSIZE | PARENB);

	//波特率
	switch (m_dwBaudRate)
	{
		case 2400:
		{
			cfsetispeed(&newtio, B2400);
			cfsetospeed(&newtio, B2400);
			break;
		}
		case 4800:
		{
			cfsetispeed(&newtio, B4800);
			cfsetospeed(&newtio, B4800);
			break;
		}
		case 9600:
		{
			cfsetispeed(&newtio, B9600);
			cfsetospeed(&newtio, B9600);
			break;
		}
		case 38400:
		{
			cfsetispeed(&newtio, B38400);
			cfsetospeed(&newtio, B38400);
			break;
		}
		case 19200:
		{
			cfsetispeed(&newtio, B19200);
			cfsetospeed(&newtio, B19200);
			break;
		}
		case 57600:
		{
			cfsetispeed(&newtio, B57600);
			cfsetospeed(&newtio, B57600);
			break;
		}
		case 115200:
		{
			cfsetispeed(&newtio, B115200);
			cfsetospeed(&newtio, B115200);
			break;
		}
		case 460800:
		{
			cfsetispeed(&newtio, B460800);
			cfsetospeed(&newtio, B460800);
			break;
		}
		case 921600:
		{
			cfsetispeed(&newtio, B921600);
			cfsetospeed(&newtio, B921600);
			break;
		}
		default:
		{
			printf("Unsupport %ld!\n", m_dwBaudRate);
			return -1;
		}
	}

	switch (m_Parity)
	{
		case 'O':
		{
			newtio.c_cflag |= (PARODD | PARENB); //使用奇校验不是用偶校验
			newtio.c_iflag |= INPCK;

			break;
		}
		case 'E':
		{
			newtio.c_cflag |= PARENB;
			newtio.c_cflag &= ~PARODD;     //使用偶校验
			newtio.c_iflag |= INPCK;
			break;
		}
		case 'N':
		{
			newtio.c_cflag &= ~PARENB;     //清除校验位
			newtio.c_iflag &= ~(ICRNL | INPCK | IXON | IXOFF);      //关闭奇偶校验  关闭软件流控

			break;
		}
		default:
		{
			printf("Unsupport Parity = %d!\n", m_Parity);
			return -1;
		}
	}

	switch (m_ByteSize)
	{
		case 7:
		{
			newtio.c_cflag |= CS7;
			break;
		}
		case 8:
		{
			newtio.c_cflag |= CS8;
			break;
		}
		default:
		{
			printf("Unsupport Size = %d!\n", m_ByteSize);
			return -1;
		}

	}

	switch (m_StopBits)
	{
		case 1:
		{
			newtio.c_cflag &= ~CSTOPB;
			newtio.c_cflag &= ~CRTSCTS;   //禁用硬件流控
			//newtio.c_cflag |= CRTSCTS;    //启用硬件流控
			break;
		}
		case 2:
		{
			newtio.c_cflag |= CSTOPB;
			break;
		}
		default:
		{
			printf("Unsupport StopBits = %d!\n", m_StopBits);
			return -1;
		}
	}

	newtio.c_cc[VTIME] = 0.01;
	newtio.c_cc[VMIN] = 1;//1;

	tcflush(com_fd, TCIFLUSH);

	if((tcsetattr(com_fd,TCSANOW,&newtio))!=0)//激活新配置
	{
		perror("Com tcsetattr error!");
		return -1;
	}

	return 0;
}

int CCom::WriteDat(void *pBuf, int Length, void *OtherInfo)
{
	//int Size_real = 0;

	//Size_real = write(com_fd, pBuf, Length);
	//while (Size_real < Length)
	//{
	//	int Length_Follow = Length - Size_real;
	//	Length_Follow = write(com_fd, (char *)pBuf + Size_real, Length_Follow);
	//	Size_real = Size_real + Length_Follow;
	//}

	int Size_real = 0;
	Size_real = write(com_fd, pBuf, Length);
	return Size_real;
}


int CCom::ReadDat(void *pBuf, int Length, void *OtherInfo)
{
	if (pBuf == NULL)
	{
		return -1;
	}

	int ret = 0;
	int nfds = 0;

	struct timeval tv;

	tv.tv_sec = (int)(m_Overtime / 1000.0);
	tv.tv_usec = m_Overtime % 1000;

	int done_lens = 0;
	int read_lens = Length;

	fd_set read_fd_set;
	FD_ZERO(&read_fd_set);
	FD_SET(com_fd, &read_fd_set);

	nfds = select(com_fd + 1, &read_fd_set, NULL, NULL, &tv);

	if (nfds == 0)
	{
		return 0;
	}
	else
	{
		usleep(1);
		for (int i = 0; i < 4096; i++)
		{
			int real_lens = read(com_fd, (unsigned char *)pBuf + done_lens, read_lens);

			if (real_lens == -1)
			{
				continue;
			}

			done_lens = done_lens + real_lens;
			read_lens = read_lens - real_lens;
		}
	}

	return done_lens;
}

int CCom::CloseDev(void *ParaInfo)
{
	close(com_fd);
	return 0;
}

int CCom::ResetDev(void *ParaInfo)
{
	return 0;
}
