#ifndef CERT_TRANS_UTIL_COMPARE_H_
#define CERT_TRANS_UTIL_COMPARE_H_

#include <ctype.h>
#include <algorithm>
#include <string>

namespace cert_trans {


template <class T>
struct ci_less;


template <>
struct ci_less<int> {
  bool operator()(int lhs, int rhs) const {
    return tolower(lhs) < tolower(rhs);
  }
};


template <>
struct ci_less<std::string> {
  bool operator()(const std::string& lhs, const std::string& rhs) const {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                        rhs.end(), ci_less<int>());
  }
};


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_COMPARE_H_
