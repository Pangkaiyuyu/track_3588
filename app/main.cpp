#include "project_common.h"

#include "Process_global.h"
#include "Process_ImgDistribution.h"
#include "Process_SelfCheck.h"
#include "Process_SelfTest.h"

#include "Process_CmpDistribution.h"
#include "Process_ImgDistribution.h"
#include "Process_MPP.h"
#include "Process_CsrTrack.h"

using namespace std;

FIFO<img_pack *> queue_img_src_to_vpss(20);
FIFO<img_pack *> queue_img_src_to_tracker(20);//new add
FIFO<stream_pack *> queue_stream_to_out(100);

CUdp * m_udp_p2p;
CCom * m_com_p2p;
CPCIE * m_pcie_p2p;

int flag_end = 0;//flag_end: 1为终止全部线程
int roi_1,roi_2,roi_3,roi_4 = 0;
bool b_track_init = true;

int main(int argc, char **argv)
{
	//test_mpp(argc, argv);
	//return 0;

	Init_Log();

	int s32Ret = 0;

	LOG(INFO) << "====== SOFTWARE_BUILDTM :  " << BH_SOFTWARE_DATE << "  " << BH_SOFTWARE_TIME << "  ======";
	LOG(INFO) << "====== SOFTWARE_VERSION :  " << BH_SOFTWARE_VERSION << "  ============";

	s32Ret = Interface_Init();

	if (s32Ret == -1)
	{
		LOG(WARNING) << "Interface_Init Failed!";
		return 0;
	}

	usleep(100 * 1000);

	s32Ret = mpp_init();

	if (s32Ret == -1)
	{
		LOG(WARNING) << "mpp_init Failed!";
		return 0;
	}


	CProcess_ImgDistribution *B_Process_ImgDistribution;
	B_Process_ImgDistribution = new CProcess_ImgDistribution();
	B_Process_ImgDistribution->Config_Thread(80, true, 1);

	CProcess_SelfCheck *B_Process_SelfCheck;
	B_Process_SelfCheck = new CProcess_SelfCheck();
	B_Process_SelfCheck->Config_Thread(10, true, 0);

	CProcess_SelfTest *B_Process_SelfTest;
	B_Process_SelfTest = new CProcess_SelfTest();
	B_Process_SelfTest->Config_Thread(95, true, 2);

	CProcess_CmpDistribution *B_Process_CmpDistribution;
	B_Process_CmpDistribution = new CProcess_CmpDistribution();
	B_Process_CmpDistribution->Config_Thread(80, true, 3);

	CProcess_CsrTrack *B_Process_CsrTrack;
	B_Process_CsrTrack = new CProcess_CsrTrack();
	B_Process_CsrTrack->Config_Thread(50, true, 5);//我觉得应该80，要不然人家编码都发送了，这边还没跟踪上呢

	B_Process_CmpDistribution->Start_Thread();
	usleep(100 * 1000);

	B_Process_SelfCheck->Start_Thread();
	usleep(100 * 1000);

	B_Process_ImgDistribution->Start_Thread();
	usleep(100 * 1000);

	B_Process_CsrTrack->Start_Thread();
	usleep(100 * 1000);

#if SELF_TEST
	B_Process_SelfTest->Start_Thread();
#endif

	usleep(2000 * 1000);

	LOG(INFO) << "---------------Program Running ALL !!--------------------------";

	B_Process_SelfCheck->Wait_Thread_Exit();
	LOG(INFO) << "B_Process_SelfCheck exit";

	B_Process_CmpDistribution->Wait_Thread_Exit();
	LOG(INFO) << "B_Process_CmpDistribution exit";

	B_Process_ImgDistribution->Wait_Thread_Exit();
	LOG(INFO) << "B_Process_ImgDistribution exit";

	B_Process_CsrTrack->Wait_Thread_Exit();
	LOG(INFO) << "B_Process_CsrTrack exit";

	LOG(INFO) << "--------------- Program Ready To Stop  !!--------------------------";

	LOG(INFO) << "Interface_Quit...";
	Interface_Quit();

	queue_img_src_to_vpss.Reset();
	queue_img_src_to_tracker.Reset();
	queue_stream_to_out.Reset();

	LOG(INFO) << "Program End  !!";

	Quit_Log();

	return 0;
}




