#ifndef OPENSSL_UTIL_H
#define OPENSSL_UTIL_H

#include <glog/logging.h>
#include <openssl/err.h>
#include <string>

namespace util {

// Dump all OpenSSL errors and clear the stack. Use like so:
// LOG(level) << DumpOpenSSLErrorStack();
// to avoid doing unnecessary work when the requested logging level
// is not enabled. (However note that in this case the error stack
// should be cleared separately.)
//
// Call ERR_load_crypto_strings()/ERR_load_ssl_strings() first to get
// human-readable strings.
std::string DumpOpenSSLErrorStack();

void ClearOpenSSLErrors();

std::string ReadBIO(BIO* bio);

}  // namespace util

// Convenience macro to help automatically clear the stack regardless of
// whether the requested logging level is high enough.
// Defined as macro so that logging happens locally where the error occurred.
#define LOG_OPENSSL_ERRORS(severity)                \
  do {                                              \
    LOG(severity) << util::DumpOpenSSLErrorStack(); \
    util::ClearOpenSSLErrors();                     \
  } while (0);

#endif  // OPENSSL_UTIL_H
