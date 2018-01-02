#include "util/testing.h"

#include <event2/thread.h>
#include <evhtp.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "config.h"

DEFINE_string(test_srcdir, TEST_SRCDIR, "top-level of the source tree");

namespace cert_trans {
namespace test {

void InitTesting(const char* name, int* argc, char*** argv,
                 bool remove_flags) {
  ::testing::InitGoogleTest(argc, *argv);
  google::ParseCommandLineFlags(argc, argv, remove_flags);
  google::InitGoogleLogging(name);
  google::InstallFailureSignalHandler();
  evthread_use_pthreads();

  // Set-up OpenSSL for multithreaded use:
  evhtp_ssl_use_threads();

  OpenSSL_add_all_algorithms();
  ERR_load_BIO_strings();
  ERR_load_crypto_strings();
  SSL_load_error_strings();
  SSL_library_init();
}

}  // namespace test
}  // namespace cert_trans
