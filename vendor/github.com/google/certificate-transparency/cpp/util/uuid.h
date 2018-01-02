#ifndef CERT_TRANS_UTIL_UUID_H_
#define CERT_TRANS_UTIL_UUID_H_

#include <string>

namespace cert_trans {

// Generates a type 4 (128bit random) UUID:
std::string UUID4();

}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_UUID_H_
