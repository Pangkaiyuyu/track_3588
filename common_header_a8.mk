ifndef $(COMMON_HEADER_MK)
COMMON_HEADER_MK = 1
CROSS:=aarch64-linux-gnu
#CROSS:=

ifeq ($(MODULE_ROOT), )
ifeq ($(BASE_DIR), )
BASE_DIR=$(CURDIR)
endif
MODULE_LIB_BASE_DIR=$(BASE_DIR)/lib
MODULE_OBJ_BASE_DIR=$(MODULE_LIB_BASE_DIR)/obj
MODULE_EXE_BASE_DIR=$(BASE_DIR)/bin
else
MODULE_LIB_BASE_DIR=$(MODULE_ROOT)/lib
MODULE_OBJ_BASE_DIR=$(MODULE_LIB_BASE_DIR)/obj
MODULE_EXE_BASE_DIR=$(MODULE_ROOT)/bin
endif

ifeq ($(MODULE), )
MODULE=$(notdir $(CURDIR))
endif

MODULE_LIB_DIR = $(MODULE_LIB_BASE_DIR)
MODULE_OBJ_DIR=$(MODULE_OBJ_BASE_DIR)/$(MODULE)
MODULE_EXE_DIR=$(MODULE_EXE_BASE_DIR)

MODULE_LIB = $(MODULE_LIB_DIR)/$(MODULE).a
MODULE_EXE = $(MODULE_EXE_DIR)/$(MODULE).out

CXX = $(CROSS)-g++
CC  = $(CROSS)-gcc
AR  = $(CROSS)-ar
STRIP  = $(CROSS)-strip

# PUB_DIR = $(CURDIR)
PUB_DIR = ..
$(info BASE_DIR is $(BASE_DIR))

INC_PATH = -I$(PUB_DIR)/include -I$(PUB_DIR)/include/opencv3410 -I$(PUB_DIR)/include/rockchip -I$(shell pwd)
$(info INC_PATH is $(INC_PATH))
LIB_PATH = -L$(PUB_DIR)/lib -L$(PUB_DIR)/lib/glog -L$(PUB_DIR)/lib/opencv3410 -L$(PUB_DIR)/lib/rockchip -L$(PUB_DIR)/lib/ffmpeg 

LIB = -fPIC -lrt -lpthread -lm -ldl -lz -Wno-date-time -fno-aggressive-loop-optimizations -Wl,-z,relro,-z,now,-z,noexecstack -Wl,--gc-sections
LIB += -lglog
LIB += -lopencv_world
LIB += -lrockchip_mpp
LIB += -lavcodec -lavformat -lavutil -lswscale -lx264 -lswresample


COMM_CFLAGS = -c -Wall -Warray-bounds -D_GNU_SOURCE  -fomit-frame-pointer -ffunction-sections -fdata-sections -O2 -g

WARNNING_IGNORED_FLAG = -Wno-deprecated-declarations -Wno-unused-but-set-variable  -Wno-unused-variable -Wno-unused-function -Wno-unused-local-typedefs -Wno-unused-result

CC_OPTS  = $(COMM_CFLAGS) $(WARNNING_IGNORED_FLAG)
CXX_OPTS = $(COMM_CFLAGS) $(WARNNING_IGNORED_FLAG)

AR_OPTS = -rc
LD_OPTS = $(LIB_PATH) $(LIB)

endif # ifndef $(COMMON_HEADER_MK)




