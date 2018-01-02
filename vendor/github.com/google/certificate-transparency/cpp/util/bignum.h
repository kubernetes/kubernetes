#ifndef CERT_TRANS_UTIL_BIGNUM_H_
#define CERT_TRANS_UTIL_BIGNUM_H_

#include <openssl/bn.h>
#include <cassert>

namespace cert_trans {


class BigNum {
 public:
  BigNum();
  explicit BigNum(int64_t w);
  BigNum(const BigNum& other);
  ~BigNum();

  const BIGNUM* bn() const;
  BIGNUM* bn();

  int num_bits() const;

  BigNum& operator-=(const BigNum& n);
  BigNum& operator-=(int64_t n);
  BigNum& operator+=(const BigNum& n);
  BigNum& operator+=(int64_t n);
  BigNum& operator<<=(int n);
  BigNum& operator>>=(int n);

 private:
  BIGNUM bn_;

  friend bool operator==(const BigNum& a, const BigNum& b);
  friend bool operator==(const BigNum& a, int64_t b);
  friend bool operator<(const BigNum& a, const BigNum& b);
  friend bool operator<(const BigNum& a, int64_t b);
  friend bool operator>(const BigNum& a, const BigNum& b);
  friend bool operator>(const BigNum& a, int64_t b);
};


inline const BIGNUM* BigNum::bn() const {
  return &bn_;
}


inline BIGNUM* BigNum::bn() {
  return &bn_;
}


inline int BigNum::num_bits() const {
  return BN_num_bits(&bn_);
}


inline BigNum& BigNum::operator-=(const BigNum& n) {
  assert(BN_sub(&bn_, &bn_, &n.bn_) == 1);
  return *this;
}


inline BigNum& BigNum::operator-=(int64_t n) {
  assert(BN_sub_word(&bn_, n) == 1);
  return *this;
}


inline BigNum& BigNum::operator+=(const BigNum& n) {
  assert(BN_add(&bn_, &bn_, &n.bn_) == 1);
  return *this;
}


inline BigNum& BigNum::operator+=(int64_t n) {
  assert(BN_add_word(&bn_, n) == 1);
  return *this;
}


inline BigNum& BigNum::operator<<=(int n) {
  assert(BN_lshift(&bn_, &bn_, n) == 1);
  return *this;
}


inline BigNum& BigNum::operator>>=(int n) {
  assert(BN_rshift(&bn_, &bn_, n) == 1);
  return *this;
}


template <typename T, typename U>
inline BigNum operator+(const T& a, const U& b) {
  BigNum r(a);
  r += b;
  return r;
}


template <typename T, typename U>
inline BigNum operator-(const T& a, const U& b) {
  BigNum r(a);
  r -= b;
  return r;
}


inline BigNum operator<<(const BigNum& a, int n) {
  BigNum r(a);
  r <<= n;
  return r;
}


inline BigNum operator>>(const BigNum& a, int n) {
  BigNum r(a);
  r >>= n;
  return r;
}


inline bool operator==(const BigNum& a, const BigNum& b) {
  return BN_cmp(&a.bn_, &b.bn_) == 0;
}


namespace internal {


inline const BigNum& AsBigNum(const BigNum& n) {
  return n;
}


inline BigNum AsBigNum(int64_t n) {
  return BigNum(n);
}


}  // namespace internal


template <typename T, typename U>
inline bool operator==(const T& a, const U& b) {
  return internal::AsBigNum(a) == internal::AsBigNum(b);
}


inline bool operator<(const BigNum& a, const BigNum& b) {
  return BN_cmp(&a.bn_, &b.bn_) < 0;
}


template <typename T, typename U>
inline bool operator<(const T& a, const U& b) {
  return internal::AsBigNum(a) < internal::AsBigNum(b);
}


inline bool operator>(const BigNum& a, const BigNum& b) {
  return BN_cmp(&a.bn_, &b.bn_) > 0;
}


template <typename T, typename U>
inline bool operator>(const T& a, const U& b) {
  return internal::AsBigNum(a) > internal::AsBigNum(b);
}


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_BIGNUM_H_
