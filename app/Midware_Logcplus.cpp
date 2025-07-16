/************************************************************************
 *FileName:  Log_cplus.cpp
 *Author:    Jin Qiuyu
 *Description: 中间件-日志管理
 ************************************************************************/

#include "Midware_Logcplus.h"

void Init_Log(void)
{
	DIR *dirptr = NULL;
	dirptr = opendir("/home/log/");

	if (dirptr == NULL)
	{
		if (mkdir("/home/log/", 0755) < 0) //0755为八进制数
		{
			printf("error mkdir /home/log\n");
			_exit(0);
		}
	}
	else
	{
		closedir(dirptr);
		dirptr = NULL;
	}

	google::InitGoogleLogging("M_U204_APP");

	FLAGS_log_dir = "/home/log/";
	//FLAGS_logtostderr = true;  //设置日志消息是否转到标准输出而不是日志文件//
	FLAGS_alsologtostderr = true;  //设置日志消息除了日志文件之外是否去标准输出//
	FLAGS_colorlogtostderr = false;  //设置记录到标准输出的颜色消息（如果终端支持）
	FLAGS_log_prefix = true;  //设置日志前缀是否应该添加到每行输出
	FLAGS_logbufsecs = 10;  //设置可以缓冲日志的最大秒数，0指实时输出
	FLAGS_max_log_size = 1;  //设置最大日志文件大小（以MB为单位）
	FLAGS_stop_logging_if_full_disk = true;  //设置是否在磁盘已满时避免日志记录到磁盘
}

void Quit_Log(void)
{
	google::ShutdownGoogleLogging();
}

void printDebug(void)
{
	LOG(INFO) << "info test";  //输出一个Info日志
	LOG(WARNING) << "warning test";  //输出一个Warning日志
	LOG(ERROR) << "error test";  //输出一个Error日志
	LOG(FATAL) << "fatal test";  //输出一个Fatal日志，这是最严重的日志并且输出之后会中止程序
	//LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";  //当条件满足时输出日志
	//LOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";  //google::COUNTER 记录该语句被执行次数，从1开始，在第一次运行输出日志之后，每隔 10 次再输出一次日志信息
	//LOG_IF_EVERY_N(INFO, (size > 1024), 10) << "Got the " << google::COUNTER << "th big cookie";  //上述两者的结合，不过要注意，是先每隔 10 次去判断条件是否满足，如果滞则输出日志；而不是当满足某条件的情况下，每隔 10 次输出一次日志信息
	//LOG_FIRST_N(INFO, 20) << "Got the " << google::COUNTER << "th cookie";  //当此语句执行的前 20 次都输出日志，然后不再输出
}