/************************************************************************
 *FileName:  Midware_Fifo.h
 *Author:    Jin Qiuyu
 *Version:   2.0
 *Date:      2020-03-08
 *Description: 中间件-FIFO类
 ************************************************************************/
#ifndef _MIDWARE_FIFO_H
#define _MIDWARE_FIFO_H

#include "project_common.h"

#include <queue>

using namespace std;

template<class T>
class FIFO
{
public:
	FIFO(unsigned int nDeepth = 20)
	{
		m_maxdepth = nDeepth;
		pthread_mutex_init(&m_mutex, NULL);
	}

	~FIFO(void)
	{
		pthread_mutex_destroy(&m_mutex);
	}

	void WriteDat(T pMem)
	{
		T pMem_abandon = NULL;

		pthread_mutex_lock(&m_mutex);
		if (m_queue.size() > m_maxdepth)
		{
			if (!m_queue.empty())
			{
				pMem_abandon = m_queue.front();
				delete pMem_abandon;
				m_queue.pop();
			}
		}
		m_queue.push(pMem);
		pthread_mutex_unlock(&m_mutex);
	}

	T ReadDat(void)
	{
		T pMem = NULL;

		pthread_mutex_lock(&m_mutex);
		if (!m_queue.empty())
		{
			pMem = m_queue.front();
			m_queue.pop();
		}
		pthread_mutex_unlock(&m_mutex);

		return pMem;
	}

	T ReadDatAll(void)
	{
		T pMem = NULL;

		pthread_mutex_lock(&m_mutex);

		while (m_queue.size() > 1)
		{
			pMem = m_queue.front();
			delete pMem;
			pMem = NULL;
			m_queue.pop();
		}

		if (!m_queue.empty())
		{
			pMem = m_queue.front();
			m_queue.pop();
		}

		pthread_mutex_unlock(&m_mutex);

		return pMem;
	}

	void Reset(void)
	{
		T pMem_abandon = NULL;

		pthread_mutex_lock(&m_mutex);
		while (!m_queue.empty())
		{
			pMem_abandon = m_queue.front();
			delete pMem_abandon;
			m_queue.pop();
		}
		pthread_mutex_unlock(&m_mutex);
	}

	unsigned int GetLenth(void)
	{
		return m_queue.size();
	}

private:

	unsigned int m_maxdepth;
	queue <T> m_queue;
	pthread_mutex_t m_mutex;
};


#endif // _MIDWARE_FIFO_H
