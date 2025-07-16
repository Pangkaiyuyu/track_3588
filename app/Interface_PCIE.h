/************************************************************************
 *FileName:  Interface_PCIE.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2021-9-27
 *Description: 接口子类
 ************************************************************************/
#ifndef _INTERFACE_PCIE_H
#define _INTERFACE_PCIE_H

#include "project_common.h"

#define PCI_MAX_BUS 255
#define PCI_MAX_DEV 31
#define PCI_MAX_FUN 7

#define PCICFG_REG_VID 0x00
#define PCICFG_REG_DID 0x02
#define PCICFG_REG_CMD 0x04
#define PCICFG_REG_RID 0x08
#define PCICFG_REG_BAR0 0x10

 //PCIE offset bit
#define PCIE_RESET 0x0300

#define PCIE_MEMADDR_H_R 0x0100
#define PCIE_MEMADDR_L_R 0x0104
#define PCIE_PACKLENS_R 0x0108
#define PCIE_PACKLENS_R_VIEW 0x010C
#define PCIE_START_TRANS_R 0x0110

#define PCIE_MEMADDR_H_W 0x0200
#define PCIE_MEMADDR_L_W 0x0204
#define PCIE_TLP_W 0x020C
#define PCIE_PACKLENS_W 0x0208
#define PCIE_START_TRANS_W 0x0210


#define PCIE_TLP_R_MOD 0xf000
#define PCIE_TLP_W_MOD 0xf004


#define PCIE_DONE_TRANS_R 0x0114
#define PCIE_DONE_CLEAR_TRANS_R 0x011C

#define PCIE_DONE_TRANS_W 0x0214
#define PCIE_DONE_CLEAR_TRANS_W 0x021C


#define PCIE_CHANGE_CH 0x11d0

//#define IMAGE_HEIGHT_SOURSE 512
//#define IMAGE_WIDTH_SOURSE  640

//#define IMAGE_HEIGHT_SOURSE 1080
//#define IMAGE_WIDTH_SOURSE  1920

class CPCIE
{
public:
	CPCIE();
	~CPCIE();

	int OpenDev(void *ParaInfo);
	int Config(void *ParaInfo);	
	int CloseDev(void *ParaInfo);
	int ResetDev(void *ParaInfo);
	int WriteDat(void *pBuf, int Length, void *OtherInfo);
	int ReadDat(void *pBuf, int Length, void *OtherInfo);

	int Init();//初始化：打开并配置串口//
	int Start_Test();

	int choose(int ch);

private:

	uint32_t gpdrv_getcount();
	uint64_t virtual_to_physical(uint64_t addr);
	int get_pid(const char* process_name);

	int cfg_handle;
	int fd_bar0;
	int fd_mem_read;

	unsigned char *bar0_map_base;
	unsigned char *map_mem_base_r1; //map adddresss to load data

	uint32_t addr_bar0;
	unsigned long addr_base_r1;

};

#endif // _INTERFACE_PCIE_H
