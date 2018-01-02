#ifndef CERT_TRANS_UTIL_READ_PRIVATE_KEY_H_
#define CERT_TRANS_UTIL_READ_PRIVATE_KEY_H_

#include <openssl/evp.h>
#include <string>

#include "util/statusor.h"

namespace cert_trans {


util::StatusOr<EVP_PKEY*> ReadPrivateKey(const std::string& file);

util::StatusOr<EVP_PKEY*> ReadPublicKey(const std::string& file);


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_READ_PRIVATE_KEY_H_
