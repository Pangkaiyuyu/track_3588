/************************************************************************
 *FileName:  Process_SelfCheck.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 应用层-自检类
 ************************************************************************/

#include "Process_SelfCheck.h"

CProcess_SelfCheck::CProcess_SelfCheck(void)
{
}

CProcess_SelfCheck::~CProcess_SelfCheck(void)
{
}

unsigned long long CProcess_SelfCheck::Deal(void)
{
	LOG(INFO) << "CProcess_SelfCheck start";

	unsigned int pid = get_pid("m_u204_app.out");

	LOG(INFO) << "pid = " << pid;

	int print_fre = 0;

	float m_cpu_cur = 0.0;
	float m_res_mem = 0.0;
	float m_vir_mem = 0.0;

	while (flag_end != 1)
	{
		m_cpu_cur = get_proc_cpu(pid);
		m_res_mem = get_proc_mem(pid) / 1024.0;
		m_vir_mem = get_proc_virtualmem(pid) / 1024.0;

		print_fre++;

		if (print_fre == 100)
		{
			LOG(INFO) << "CPU :" << fixed << setprecision(2) << m_cpu_cur << "%";
			LOG(INFO) << "Res mem  :" << fixed << setprecision(2) << m_res_mem;
			LOG(INFO) << "Vir mem  :" << fixed << setprecision(2) << m_vir_mem;
			print_fre = 0;
		}

		usleep(1000 * 1000);
	}

	return 0;
}

typedef struct {
	unsigned long user;
	unsigned long nice;
	unsigned long system;
	unsigned long idle;
}Total_Cpu_Occupy_t;


typedef struct {
	unsigned int pid;
	unsigned long utime;  //user time
	unsigned long stime;  //kernel time
	unsigned long cutime; //all user time
	unsigned long cstime; //all dead time
}Proc_Cpu_Occupy_t;

#define VMRSS_LINE 22
#define VMSIZE_LINE 17
#define PROCESS_ITEM 14
#define TMP_BUF_LEN 256

const char* CProcess_SelfCheck::get_items(const char*buffer, unsigned int item) 
{

	const char *p = buffer;

	int len = strlen(buffer);
	unsigned int count = 0;

	for (int i = 0; i < len; i++) {
		if (' ' == *p) {
			count++;
			if (count == item - 1) {
				p++;
				break;
			}
		}
		p++;
	}

	return p;
}

unsigned long CProcess_SelfCheck::get_cpu_total_occupy() 
{

	FILE *fd;
	char buff[1024] = { 0 };
	Total_Cpu_Occupy_t t;

	fd = fopen("/proc/stat", "r");
	if (nullptr == fd) 
	{
		return 0;
	}

	if (fgets(buff, sizeof(buff), fd) == NULL) 
	{
		return 0;
	}

	char name[64] = { 0 };
	sscanf(buff, "%s %ld %ld %ld %ld", name, &t.user, &t.nice, &t.system, &t.idle);
	fclose(fd);

	return (t.user + t.nice + t.system + t.idle);
}

unsigned long CProcess_SelfCheck::get_cpu_proc_occupy(unsigned int pid) 
{

	char file_name[64] = { 0 };
	Proc_Cpu_Occupy_t t;
	FILE *fd;
	char line_buff[1024] = { 0 };
	sprintf(file_name, "/proc/%d/stat", pid);

	fd = fopen(file_name, "r");
	if (nullptr == fd) {
		return 0;
	}

	if (fgets(line_buff, sizeof(line_buff), fd) == NULL)
	{
		return 0;
	}

	sscanf(line_buff, "%u", &t.pid);
	const char *q = get_items(line_buff, PROCESS_ITEM);
	sscanf(q, "%ld %ld %ld %ld", &t.utime, &t.stime, &t.cutime, &t.cstime);
	fclose(fd);

	return (t.utime + t.stime + t.cutime + t.cstime);
}

float CProcess_SelfCheck::get_proc_cpu(unsigned int pid) 
{

	unsigned long totalcputime1, totalcputime2;
	unsigned long procputime1, procputime2;

	totalcputime1 = get_cpu_total_occupy();
	procputime1 = get_cpu_proc_occupy(pid);

	usleep(200*1000);

	totalcputime2 = get_cpu_total_occupy();
	procputime2 = get_cpu_proc_occupy(pid);

	float pcpu = 0.0;
	if (0 != totalcputime2 - totalcputime1) {
		pcpu = 100.0 * (procputime2 - procputime1) / (totalcputime2 - totalcputime1);
	}

	return pcpu;
}

unsigned int CProcess_SelfCheck::get_proc_mem(unsigned int pid) 
{

	char file_name[64] = { 0 };
	FILE *fd;
	char line_buff[512] = { 0 };
	sprintf(file_name, "/proc/%d/status", pid);

	fd = fopen(file_name, "r");
	if (nullptr == fd) {
		return 0;
	}

	char name[64];
	int vmrss;
	for (int i = 0; i < VMRSS_LINE - 1; i++) 
	{
		if (fgets(line_buff, sizeof(line_buff), fd) == NULL)
		{
			return 0;
		}
	}

	if (fgets(line_buff, sizeof(line_buff), fd) == NULL)
	{
		return 0;
	}

	sscanf(line_buff, "%s %d", name, &vmrss);
	fclose(fd);

	return vmrss;
}

unsigned int CProcess_SelfCheck::get_proc_virtualmem(unsigned int pid) 
{

	char file_name[64] = { 0 };
	FILE *fd;
	char line_buff[512] = { 0 };
	sprintf(file_name, "/proc/%d/status", pid);

	fd = fopen(file_name, "r");
	if (nullptr == fd) {
		return 0;
	}

	char name[64];
	int vmsize;
	for (int i = 0; i < VMSIZE_LINE - 1; i++) 
	{
		if (fgets(line_buff, sizeof(line_buff), fd) == NULL)
		{
			return 0;
		}
	}

	if (fgets(line_buff, sizeof(line_buff), fd) == NULL)
	{
		return 0;
	}
	sscanf(line_buff, "%s %d", name, &vmsize);
	fclose(fd);

	return vmsize;
}

int CProcess_SelfCheck::get_pid(const char* process_name)
{
	char cmd[256];

	sprintf(cmd, "pgrep %s", process_name);
	
	FILE *pstr = popen(cmd, "r");

	if (pstr == nullptr) 
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

int Interface_Init()
{
	int ret = 0;

	m_pcie_p2p = new CPCIE();

	ret = m_pcie_p2p->Init();

	if (ret != 0)
	{
		LOG(WARNING) << "m_pcie_p2p init failed!";
		return -1;
	}

	m_udp_p2p = new CUdp("192.168.100.3", 10003, "192.168.100.1", 10003, 0, 0, true, 2000);

	ret = m_udp_p2p->Init();

	if (ret != 0)
	{
		LOG(WARNING) << "m_udp_p2p init failed!";
		return -1;
	}

	return 0;
}

int Interface_Quit()
{
	if (m_pcie_p2p != NULL)
	{
		m_pcie_p2p->CloseDev(NULL);
		delete m_pcie_p2p;
		m_pcie_p2p = NULL;
	}

	if (m_udp_p2p != NULL)
	{
		m_udp_p2p->CloseDev(NULL);
		delete m_udp_p2p;
		m_udp_p2p = NULL;
	}

	if (m_com_p2p != NULL)
	{
		m_com_p2p->CloseDev(NULL);
		delete m_com_p2p;
		m_com_p2p = NULL;
	}

	return 0;
}

int fileNameFilter(const struct dirent *cur)
{
	std::string str(cur->d_name);
	if ((str.find(".bin") != std::string::npos))
	{
		return 1;
	}
	return 0;
}

int Sys_Init()
{
	
	return 0;
}
