#include "util/init.h"

#include <event2/thread.h>
#include <evhtp.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <unistd.h>
#include <string>

#include "config.h"
#include "log/ct_extensions.h"
#include "proto/cert_serializer.h"
#include "version.h"

using std::string;

namespace util {


namespace {


void LibEventLog(int severity, const char* msg) {
  const string msg_s(msg);
  switch (severity) {
    case EVENT_LOG_DEBUG:
      VLOG(1) << msg_s;
      break;
    case EVENT_LOG_MSG:
      LOG(INFO) << msg_s;
      break;
    case EVENT_LOG_WARN:
      LOG(WARNING) << msg_s;
      break;
    case EVENT_LOG_ERR:
      LOG(ERROR) << msg_s;
      break;
    default:
      LOG(ERROR) << "LibEvent(?): " << msg_s;
      break;
  }
}


}  // namespace


void InitCT(int* argc, char** argv[]) {
  google::SetVersionString(cert_trans::kBuildVersion);
  google::ParseCommandLineFlags(argc, argv, true);
  google::InitGoogleLogging(*argv[0]);
  google::InstallFailureSignalHandler();

  event_set_log_callback(&LibEventLog);

  evthread_use_pthreads();
  // Set-up OpenSSL for multithreaded use:
  evhtp_ssl_use_threads();

  OpenSSL_add_all_algorithms();
  ERR_load_BIO_strings();
  ERR_load_crypto_strings();
  SSL_load_error_strings();
  SSL_library_init();

  cert_trans::LoadCtExtensions();

  LOG(INFO) << "Build version: " << google::VersionString();
#ifdef ENABLE_HARDENING
  LOG(INFO) << "Binary built with hardening enabled.";
#else
  LOG(WARNING) << "Binary built with hardening DISABLED.";
#endif
}


}  // namespace util
