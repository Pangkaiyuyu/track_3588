ifndef $(COMMON_FOOTER_MK)
COMMON_FOOTER_MK = 1

#MODULE_FILES = $(subst ./, , $(foreach dir,.,$(wildcard $(dir)/*.c $(dir)/*.cpp)) )
MY_FILES_SUFFIX := %.cpp %.c
rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
MY_ALL_FILES := $(foreach src_path,./, $(call rwildcard,$(src_path),*.*) )
MY_ALL_FILES := $(MY_ALL_FILES:./%=%)
MODULE_FILES := $(filter $(MY_FILES_SUFFIX),$(MY_ALL_FILES))
MODULE_FILES := $(subst /./,, $(MODULE_FILES))
MODULE_OBJS   = $(subst .c,.o, $(subst .cpp,.o, $(MODULE_FILES)))
MODULE_OBJS_ABS := $(MODULE_OBJS:%=$(MODULE_OBJ_DIR)/%)

lib : $(MODULE_LIB)

$(MODULE_LIB) : $(MODULE_OBJS)
	@echo \# $(MODULE): $(CFG_CHIP_TYPE): $(CFG_BOARD_TYPE): Creating archive $(MODULE_LIB)
	#$(AR)	$(AR_OPTS) $(MODULE_LIB) $(MODULE_OBJ_DIR)/*.o
	$(AR)	$(AR_OPTS) $(MODULE_LIB) $(MODULE_OBJS_ABS)

.c.o:
	@echo \# $(MODULE): $(CFG_CHIP_TYPE): $(CFG_BOARD_TYPE): Compiling $<
	$(CC) $(CC_OPTS) $(INC_PATH) $(MODULE_INCLUDE) -o$(MODULE_OBJ_DIR)/$@ $<

.cpp.o:
	@echo \# $(MODULE): $(CFG_CHIP_TYPE): $(CFG_BOARD_TYPE): Compiling $<
	$(CXX) $(CXX_OPTS) $(INC_PATH) $(MODULE_INCLUDE) -o$(MODULE_OBJ_DIR)/$@ $<

env:
	@echo \# $(MODULE): $(CFG_CHIP_TYPE): $(CFG_BOARD_TYPE): Making Directories, if not already created..
	-mkdir -p $(MODULE_LIB_DIR)
	-mkdir -p $(MODULE_OBJ_DIR)
	-mkdir -p $(MODULE_EXE_DIR)

exe:
	@echo \# $(MODULE): $(CFG_CHIP_TYPE): $(CFG_BOARD_TYPE): Linking
	$(CXX) -Wl,--start-group  $(LD_OPTS) $(MODULE_LIBS) $(MODULE_INCLUDE) -Wl,--end-group -o$(MODULE_EXE)
	@echo \# Final executable $(MODULE_EXE) !!!
	@echo \#

strip:
	@echo \#
	@echo \# $(MODULE): $(CFG_CHIP_TYPE): $(CFG_BOARD_TYPE): Striping....
	@echo \#
	$(STRIP) $(MODULE_EXE)
	@echo \# Final executable $(MODULE_EXE) strip end !!!
	@echo \#

clean:
	@echo \# $(MODULE): $(CFG_CHIP_TYPE): $(CFG_BOARD_TYPE): Deleting temporary files
	@-rm -f $(MODULE_LIB)
	@-find $(MODULE_OBJ_DIR) -name "*.o" -delete
	@-rm -f $(MODULE_EXE)


endif # ifndef $(COMMON_FOOTER_MK)





