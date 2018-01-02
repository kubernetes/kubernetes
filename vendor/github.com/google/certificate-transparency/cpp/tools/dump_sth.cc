#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iostream>

#include "proto/ct.pb.h"
#include "version.h"

using std::cout;
using std::endl;
using std::ifstream;

namespace {


void DumpSth(const char* filename) {
  ifstream input(filename);
  ct::SignedTreeHead pb;
  CHECK(pb.ParseFromIstream(&input));

  cout << pb.DebugString() << endl;
}


}  // namespace


int main(int argc, char* argv[]) {
  google::SetVersionString(cert_trans::kBuildVersion);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Build version: " << google::VersionString();

  for (int i = 1; i < argc; ++i)
    DumpSth(argv[i]);

  return 0;
}
