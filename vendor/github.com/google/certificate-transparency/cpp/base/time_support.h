#ifndef CERT_TRANS_BASE_TIME_SUPPORT_H_
#define CERT_TRANS_BASE_TIME_SUPPORT_H_

namespace cert_trans {


static const int64_t kNumMillisPerSecond = 1000LL;

static const int64_t kNumMicrosPerMilli = 1000LL;
static const int64_t kNumMicrosPerSecond =
    kNumMillisPerSecond * kNumMicrosPerMilli;


}  // namespace cert_trans

#endif  // CERT_TRANS_BASE_TIME_SUPPORT_H_
