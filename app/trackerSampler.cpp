﻿/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "trackerPrecomp.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

/*
 *  TrackerContribSampler
 */

/*
 * Constructor
 */
TrackerContribSampler::TrackerContribSampler()
{
  blockAddTrackerSampler = false;
}

/*
 * Destructor
 */
TrackerContribSampler::~TrackerContribSampler()
{

}

void TrackerContribSampler::sampling( const Mat& image, Rect boundingBox )
{

  clearSamples();

  for ( size_t i = 0; i < samplers.size(); i++ )
  {
    std::vector<Mat> current_samples;
    samplers[i].second->sampling( image, boundingBox, current_samples );

    //push in samples all current_samples
    for ( size_t j = 0; j < current_samples.size(); j++ )
    {
      std::vector<Mat>::iterator it = samples.end();
      samples.insert( it, current_samples.at( j ) );
    }
  }

  if( !blockAddTrackerSampler )
  {
    blockAddTrackerSampler = true;
  }
}

bool TrackerContribSampler::addTrackerSamplerAlgorithm( String trackerSamplerAlgorithmType )
{
  if( blockAddTrackerSampler )
  {
    return false;
  }
  Ptr<TrackerContribSamplerAlgorithm> sampler = TrackerContribSamplerAlgorithm::create( trackerSamplerAlgorithmType );

  if (!sampler)
  {
    return false;
  }

  samplers.push_back( std::make_pair( trackerSamplerAlgorithmType, sampler ) );

  return true;
}

bool TrackerContribSampler::addTrackerSamplerAlgorithm( Ptr<TrackerContribSamplerAlgorithm>& sampler )
{
  if( blockAddTrackerSampler )
  {
    return false;
  }

  if (!sampler)
  {
    return false;
  }

  String trackerSamplerAlgorithmType = sampler->getClassName();
  samplers.push_back( std::make_pair( trackerSamplerAlgorithmType, sampler ) );

  return true;
}

const std::vector<std::pair<String, Ptr<TrackerContribSamplerAlgorithm> > >& TrackerContribSampler::getSamplers() const
{
  return samplers;
}

const std::vector<Mat>& TrackerContribSampler::getSamples() const
{
  return samples;
}

void TrackerContribSampler::clearSamples()
{
  samples.clear();
}


}}}  // namespace
