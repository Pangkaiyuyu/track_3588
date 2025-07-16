#ifndef __PROJECT_COMMON_FILE_H__
#define __PROJECT_COMMON_FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <stddef.h>
#include <dirent.h> 
#include <linux/fb.h>
#include <linux/sockios.h>
#include <linux/route.h>
#include <linux/if.h>
#include <linux/if_arp.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().
#include <sys/statfs.h>
#include <sys/utsname.h>
#include <sys/un.h>
#include <sys/prctl.h>
#include <sys/msg.h>
#include <fcntl.h>
//#include <stropts.h> //aarch64 没有该库

#include <semaphore.h>
#include <signal.h>
#include <assert.h>
#include <termios.h>   //tty
#include <ctype.h>
#include <stdarg.h>
#include <netdb.h>

#include <mqueue.h>
#include <math.h>
#include <sys/queue.h>

//#include <ft2build.h>
//#include <freetype/freetype.h>
//#include <freetype/ftglyph.h>
//#include FT_FREETYPE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace cv;

#include <fstream>  
#include <vector>
using namespace std;

#include <linux/videodev2.h>

#endif
