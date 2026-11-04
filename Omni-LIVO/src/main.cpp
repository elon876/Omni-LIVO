#include "LIVMapper.h"
#include <malloc.h>

static void configureAllocator() {
  mallopt(M_MMAP_THRESHOLD, 256 * 1024);
  mallopt(M_TRIM_THRESHOLD, 4 * 1024 * 1024);
}

int main(int argc, char **argv)
{
  configureAllocator();
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh); 
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  return 0;
}