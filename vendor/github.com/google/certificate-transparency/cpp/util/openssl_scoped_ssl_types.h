#ifndef CERT_TRANS_UTIL_OPENSSL_SSL_SCOPED_TYPES_H_
#define CERT_TRANS_UTIL_OPENSSL_SSL_SCOPED_TYPES_H_

#include <openssl/ssl.h>

#include "util/openssl_scoped_types.h"

namespace cert_trans {


using ScopedSSL = ScopedOpenSSLType<SSL, SSL_free>;
using ScopedSSL_CTX = ScopedOpenSSLType<SSL_CTX, SSL_CTX_free>;


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_OPENSSL_SSL_SCOPED_TYPES_H_
