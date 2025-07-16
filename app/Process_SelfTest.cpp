/************************************************************************
 *FileName:  Process_SelfTest.cpp
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 应用层-自测试使用
 ************************************************************************/

#include "Process_SelfTest.h"

CProcess_SelfTest::CProcess_SelfTest(void)
{
	peek_character = -1;

}

CProcess_SelfTest::~CProcess_SelfTest(void)
{

}

unsigned long long CProcess_SelfTest::Deal(void)
{
	LOG(INFO) << "CProcess_SelfTest start";

	init_keyboard();
	int n_readch = 0;
	int self_test = 0;
	int self_waibu = 0;

	while (1)
	{
		kbhit();

		n_readch = readch();

		if (n_readch == 'r')
		{
			LOG(INFO) << "Press to exit!";
			flag_end = 1;
			close_keyboard();
			break;
		}
		else
		{
			continue;
		}
		usleep(300 * 1000);
	}


	return 0;
}

void CProcess_SelfTest::init_keyboard()
{
	tcgetattr(0, &initial_settings);
	new_settings = initial_settings;
	new_settings.c_lflag |= ICANON;
	new_settings.c_lflag |= ECHO;
	new_settings.c_lflag |= ISIG;
	new_settings.c_cc[VMIN] = 1;
	new_settings.c_cc[VTIME] = 0;
	tcsetattr(0, TCSANOW, &new_settings);
}

void CProcess_SelfTest::close_keyboard()
{
	tcsetattr(0, TCSANOW, &initial_settings);
}

int CProcess_SelfTest::kbhit()
{
	unsigned char ch;
	int nread;

	if (peek_character != -1)
	{
		return 1;
	}
		
	new_settings.c_cc[VMIN] = 0;
	tcsetattr(0, TCSANOW, &new_settings);
	nread = read(0, &ch, 1);
	new_settings.c_cc[VMIN] = 1;
	tcsetattr(0, TCSANOW, &new_settings);
	if (nread == 1)
	{
		peek_character = ch;
		return 1;
	}
	return 0;
}

int CProcess_SelfTest::readch()
{
	char ch;

	if (peek_character != -1)
	{
		ch = peek_character;
		peek_character = -1;
		return ch;
	}
	read(0, &ch, 1);
	return ch;
}
