#include "util/bignum.h"

namespace cert_trans {


BigNum::BigNum() {
  BN_init(&bn_);
}


BigNum::BigNum(int64_t w) : BigNum() {
  assert(BN_set_word(&bn_, w) == 1);
}


BigNum::BigNum(const BigNum& other) : BigNum() {
  assert(BN_copy(&bn_, &other.bn_) != nullptr);
}


BigNum::~BigNum() {
  BN_free(&bn_);
}


}  // namespace cert_trans
