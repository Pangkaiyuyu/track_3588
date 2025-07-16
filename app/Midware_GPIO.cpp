/************************************************************************
 *FileName:  Midware_GPIO.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 中间件-GPIO方法
 ************************************************************************/
#include "Midware_GPIO.h"

int gpio_init(unsigned int gpio_chip_num, unsigned int gpio_offset_num, int direction)
{
	FILE *fp;
	char file_name[50];
	unsigned char buf[10];
	unsigned int gpio_num;

	gpio_num = gpio_chip_num * 8 + gpio_offset_num;

	memset(file_name, 0, sizeof(file_name));
	sprintf(file_name, "/sys/class/gpio/export");
	fp = fopen(file_name, "w");
	if (fp == NULL)
	{
		printf("Cannot open %s.\n", file_name);
		return -1;
	}

	fprintf(fp, "%d", gpio_num);
	fclose(fp);

	memset(file_name, 0, sizeof(file_name));
	sprintf(file_name, "/sys/class/gpio/gpio%d/direction", gpio_num);
	fp = fopen(file_name, "wb+");
	if (fp == NULL)
	{
		printf("Cannot open %s.\n", file_name);
		return -1;
	}


	if (direction)
	{
		strcpy((char *)buf, "out");
	}
	else
	{
		strcpy((char *)buf, "in");
	}

	fwrite(buf, sizeof(char), sizeof(buf) - 1, fp);
	fclose(fp);

	return 0;
}

int gpio_out_value(unsigned int gpio_chip_num, unsigned int gpio_offset_num,int value)
{
	FILE *fp;
	char file_name[50]; 
	unsigned char buf[10]; 
	unsigned int gpio_num;
	
	gpio_num = gpio_chip_num * 8 + gpio_offset_num;

	
	memset(file_name,0,sizeof(file_name));
	sprintf(file_name, "/sys/class/gpio/gpio%d/value", gpio_num);
	fp = fopen(file_name, "wb+");
	if (fp == NULL)
	{
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	if (value)
	{
		strcpy((char *)buf,"1");
	}
	else
	{
		strcpy((char *)buf,"0");
	}
	
	fwrite(buf, sizeof(char), sizeof(buf) - 1, fp);
	//printf("%s: gpio%d_%d = %s\n", func , gpio_chip_num, gpio_offset_num, buf);
	fclose(fp);
	return 0;
}

int gpio_in_value(unsigned int gpio_chip_num, unsigned int gpio_offset_num)
{
	FILE *fp;
	char file_name[50]; 
	unsigned char buf[10];
	unsigned int gpio_num;
	int value  =0;
	gpio_num = gpio_chip_num * 8 + gpio_offset_num;

	
	memset(file_name,0,sizeof(file_name));
	sprintf(file_name, "/sys/class/gpio/gpio%d/value", gpio_num);
	fp = fopen(file_name, "rb+");
	if (fp == NULL)
	{
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	
	
	fread(buf, sizeof(char), sizeof(buf) - 1, fp);
	value = atoi((char *)buf);
	//printf("%s: gpio%d_%d = %s\n", func , gpio_chip_num, gpio_offset_num, buf);
	fclose(fp);
	return value;
}

int gpio_set_edge(unsigned int gpio_chip_num, unsigned int gpio_offset_num,unsigned int edge)
{
    //const char dir_str[] = "none\0rising\0falling";
	char edge_buffer[12];
	
    //char ptr;
    char path[64];
    FILE * fd;
	unsigned int gpio_num;
	gpio_num = gpio_chip_num * 8 + gpio_offset_num;
	memset(edge_buffer,0,sizeof(edge_buffer));
	
    switch(edge)
    {
    case 0:
        strcpy(edge_buffer,"none");
        break;
    case 1:
        strcpy(edge_buffer,"rising");
        break;
    case 2:
       strcpy(edge_buffer,"falling");
        break;
	case 3:
       strcpy(edge_buffer,"both");
        break;

    default:
        strcpy(edge_buffer,"both");
        break;
    }

    snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/edge", gpio_num);
    fd = fopen(path, "wb+");
	if (fd == NULL)
	{
		printf("Gpio Read Error\n");
		fclose(fd);
		return false;
	}
	else
	{
	    if (fwrite(edge_buffer, strlen(edge_buffer),1,fd) != 1)
	    {
	        printf("Gpio Set Edge Error\n");
			fclose(fd);
	        return false;
	    }
		else{}
	}

    fclose(fd);
    return true;
}

