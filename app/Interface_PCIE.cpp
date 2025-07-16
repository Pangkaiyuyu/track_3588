/************************************************************************
 *FileName:  Interface_PCIE.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2021-9-27
 *Description: 接口子类
 ************************************************************************/
#include "Interface_PCIE.h"
#include <cstdio>
#include <sys/select.h>

CPCIE::CPCIE()
{
	cfg_handle = 0;

	addr_base_r1 = 0x192000000;
	map_mem_base_r1 = NULL; //map adddresss to load data
}

CPCIE::~CPCIE(void)
{

}

int CPCIE::OpenDev(void *ParaInfo)
{
	addr_bar0 = gpdrv_getcount();

	if (addr_bar0 == 0)
	{
		close(cfg_handle);
		printf("PCIE Bar0 : error\n");
		return -1;
	}

	fd_bar0 = open("/dev/mem", O_RDWR | O_SYNC);
	bar0_map_base = (unsigned char*)mmap(NULL, 0x100000, PROT_READ | PROT_WRITE, MAP_SHARED, fd_bar0, addr_bar0);

	printf("PCIE Bar0 : phy:%lx  map:%lx\n", (unsigned long)addr_bar0, (unsigned long)bar0_map_base);

	if ((unsigned long)bar0_map_base == 0xffffffffffffffff)
	{
		return -1;
	}


	fd_mem_read = open("/dev/mem", O_RDWR | O_SYNC);
	map_mem_base_r1 = (unsigned char*)mmap(NULL, 0xA00000, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem_read, addr_base_r1);

	printf("Physical Map r %lx , address:%lx\n", addr_base_r1, (unsigned long)map_mem_base_r1);

	memset(map_mem_base_r1, 0xff, 1280 * 720 * 3);

	if ((unsigned long)map_mem_base_r1 == 0xffffffffffffffff)
	{
		return -1;
	}

	for (int i = 0; i < 0xA00000; i++)
	{
		map_mem_base_r1[i] = (i) % 256;
	}


	return 0;
}

int CPCIE::Config(void *ParaInfo)
{
	*(unsigned int*)(bar0_map_base + PCIE_RESET) = 1;
	usleep(1000 * 2);
	*(unsigned int*)(bar0_map_base + PCIE_RESET) = 0;
	usleep(1000 * 2);
	*(unsigned int*)(bar0_map_base + PCIE_MEMADDR_H_R) = (int)((addr_base_r1 & 0XFFFFFFFF00000000) >> 32);
	usleep(1000 * 2);
	*(unsigned int*)(bar0_map_base + PCIE_MEMADDR_L_R) = (int)((addr_base_r1 & 0XFFFFFFFF));
	usleep(1000 * 2);

	//*(unsigned int*)(bar0_map_base + PCIE_MEMADDR_H_W) = (int)((addr_base_r1 & 0XFFFFFFFF00000000) >> 32);
	//usleep(1000 * 2);
	//*(unsigned int*)(bar0_map_base + PCIE_MEMADDR_L_W) = (int)((addr_base_r1 & 0XFFFFFFFF));
	//usleep(1000 * 2);

	*(unsigned int*)(bar0_map_base + PCIE_PACKLENS_R) = 32400;//8192
	usleep(1000 * 2);
	
	//unsigned int temp_tlp = *(unsigned int*)(bar0_map_base + PCIE_PACKLENS_R_VIEW);
	//printf("PCIE_PACKLENS_R_VIEW(0x010c) = %d\n", temp_tlp);

	//temp_tlp = *(unsigned int*)(bar0_map_base + PCIE_TLP_W);
	//printf("temp_tlp_w = %d\n", temp_tlp);
	//*(unsigned int*)(bar0_map_base + PCIE_PACKLENS_W) = 256;
	//usleep(1000 * 2);

	//*(unsigned int*)(bar0_map_base + PCIE_TLP_R_MOD) = 0;
	//usleep(1000 * 2);
	//*(unsigned int*)(bar0_map_base + PCIE_TLP_W_MOD) = 0;
	//usleep(1000 * 2);


	printf("PCIE_START_TRANS_R ...\n");
	*(unsigned int*)(bar0_map_base + PCIE_START_TRANS_R) = 1;

	return 0;
}

int CPCIE::ResetDev(void *ParaInfo)
{

	return 0;
}

int CPCIE::CloseDev(void *ParaInfo)
{
	printf("PCIE  CLOSE !!!!!!!!!!!!!!!!!!!\n");
	*(unsigned int*)(bar0_map_base + PCIE_RESET) = 1;
	usleep(1000 * 2);

	munmap(map_mem_base_r1, 0xA00000);
	munmap(bar0_map_base, 0x100000);

	close(fd_bar0);
	close(fd_mem_read);
	close(cfg_handle);

	return 0;
}

int CPCIE::WriteDat(void *pBuf, int Length, void *OtherInfo)
{
	//memcpy(map_mem_base_r1, pBuf, Length);
	*(unsigned int*)(bar0_map_base + PCIE_START_TRANS_W) = 1;//start
	int n_count_read = 0;
	int n_return = 0;

	while (1)
	{

		if (*(unsigned int*)(bar0_map_base + PCIE_DONE_TRANS_W) == 1)
		{
			//*(unsigned int*)(bar0_map_base + PCIE_DONE_CLEAR_TRANS_W) = 1;
			break;
		}
		else
		{
			n_count_read++;

			if (n_count_read > 1000)
			{
				n_return = -1;
				printf("PCIE_START_TRANS_W ffffffffff\n");
				break;
			}

			usleep(1000);
		}
	}


	return n_return;
}

int CPCIE::ReadDat(void *pBuf, int Length, void *OtherInfo)
{
	
	unsigned char * p1 = (unsigned char *)(map_mem_base_r1 + Length);
	int n_count_read = 0;
	int n_return = 0;

	while (1)
	{

		if (*(p1 + 2) == 0xaa)//0xaa
		{
			//for (int i = 0; i < 128; i++)
			//{
			//	printf("%x ", p1[i]);
			//}
			//printf("\n");

			*(p1 + 2) = 0;

			struct timeval t1, t2;

			gettimeofday(&t1, NULL);
			memcpy(pBuf, map_mem_base_r1, Length);//1080P * 2bit  56ms
			gettimeofday(&t2, NULL);

			unsigned long long m_time_now = 0;
			m_time_now = t2.tv_sec * 1000 + t2.tv_usec / 1000.0  - (t1.tv_sec * 1000 + t1.tv_usec / 1000.0);
			printf("m_time_now = %lld\n", m_time_now);
			break;
		}
		else
		{
			n_count_read++;

			if (n_count_read > 1000)
			{
				for (int i = 0; i < 128; i++)
				{
					printf("%x ", p1[i]);
				}
				printf("\n");

				n_return = -1;
				break;
			}
			usleep(1000);
		}
	}

	return n_return;
}

uint32_t CPCIE::gpdrv_getcount()
{
	int ret = 0;
	unsigned int bus, dev, fun;
	uint32_t bar0;
	int pcinodenum = 0;

	char proc_name[64];

	uint32_t data;
	uint16_t vid, did;
	uint8_t	 rid;
	uint16_t cmd;

	//open the device
	snprintf(proc_name, sizeof(proc_name), "/proc/bus/pci/0000:01/00.0");

	cfg_handle = open(proc_name, O_RDWR);

	if (cfg_handle <= 0)
	{
		printf("PCIE config File Open Failed!\n");
		return  0;
	}
	else
	{
		printf("PCIE config File Open sucessed!!!\n");
	}

	//read the register
	lseek(cfg_handle, PCICFG_REG_VID, SEEK_SET);//读取供应商ID 
	ret = read(cfg_handle, &data, sizeof(data));

	if ((data != 0xffffffff) && (data != 0))
	{
		//read the BAR0 addresss
		lseek(cfg_handle, PCICFG_REG_BAR0, SEEK_SET);
		ret = read(cfg_handle, &bar0, sizeof(bar0));
		ret = read(cfg_handle, &rid, sizeof(rid));

		lseek(cfg_handle, PCICFG_REG_CMD, SEEK_SET);   //Command Registr
		ret = read(cfg_handle, &cmd, sizeof(cmd));

		vid = data & 0xffff;    //低16位
		did = data >> 16;       //高16位

		printf("vid:%x , did:%x\n", (unsigned int)vid, (unsigned int)did);

		if ((0x0700 == vid) && (0xee10 == did))
		{
			pcinodenum++;
		}

		//printf("0000:%02x:%0x02;%02x;\n", bus, dev, fun);
		printf("rid = %d\n", rid);

		if (rid > 0)
		{
			printf("GPU DEVICE\n");
			printf("%04x:%04x(rev %02x)\n\n", (unsigned int)vid, (unsigned int)did, (unsigned int)rid);
		}
		else
		{
			printf("PCI DEVICE\n");
			printf("%04x:%04x\n", (unsigned int)vid, (unsigned int)did);

			cmd = cmd | 0x02;
			cmd = cmd & 0xfe;
			lseek(cfg_handle, PCICFG_REG_CMD, SEEK_SET);
			ret = write(cfg_handle, &cmd, sizeof(cmd));
		}
	}

	printf("pcienum = %d\n", pcinodenum);
	return bar0;
}

int CPCIE::get_pid(const char* process_name)
{
	char cmd[256];

	sprintf(cmd, "pgrep %s", process_name);

	FILE *pstr = popen(cmd, "r");

	if (pstr == NULL)
	{
		return 0;
	}

	char buff[256];
	memset(buff, 0, sizeof(buff));

	if (NULL == fgets(buff, 256, pstr))
	{
		return 0;
	}

	return atoi(buff);
}

uint64_t CPCIE::virtual_to_physical(uint64_t addr)
{
	int pid = get_pid("m_st310_app.out");
	char str_pagemap[64];
	sprintf(str_pagemap, "/proc/%d/pagemap", pid);

	int fd = open(str_pagemap, O_RDONLY);
	if (fd < 0)
	{
		printf("open %s failed!\n", str_pagemap);
		return 0;
	}
	uint64_t pagesize = getpagesize();

	uint64_t offset = (addr / pagesize) * sizeof(uint64_t);

	printf("pagesize = %lx\n", pagesize);
	printf("addr = %lx\n", addr);
	printf("offset = %lx\n", offset);


	if (lseek(fd, offset, SEEK_SET) < 0)
	{
		printf("lseek() failed!\n");
		close(fd);
		return 0;
	}
	uint64_t info;
	if (read(fd, &info, sizeof(uint64_t)) != sizeof(uint64_t))
	{
		printf("read() failed!\n");
		close(fd);
		return 0;
	}

	printf("info = %lx\n", info);

	if ((info & (((uint64_t)1) << 63)) == 0)
	{
		printf("page is not present!\n");
		close(fd);
		return 0;
	}
	uint64_t frame = info & ((((uint64_t)1) << 55) - 1);
	uint64_t phy = frame * pagesize + addr % pagesize;
	close(fd);
	return phy;
}

int CPCIE::Init()
{
	int ret = 0;

	ret = OpenDev(NULL);
	if (-1 == ret)
	{
		printf("OpenDev Failed\n");
		return -1;
	}

	ret = Config(NULL);
	if (-1 == ret)
	{
		printf("Config Failed\n");
		return -1;
	}

	return 0;
}

int CPCIE::Start_Test()
{
	*(unsigned int*)(bar0_map_base + 0x0124) = 1;
	usleep(1000 * 2);

	return 0;
}

int CPCIE::choose(int ch)
{
	*(unsigned int*)(bar0_map_base + PCIE_RESET) = 1;//reset
	usleep(1000 * 500);

	*(unsigned int*)(bar0_map_base + PCIE_CHANGE_CH) = ch;
	usleep(1000 * 2);

	*(unsigned int*)(bar0_map_base + PCIE_START_TRANS_R) = 1;
	usleep(1000 * 2);

	return 0;
}