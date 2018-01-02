#ifndef CERT_TRANS_UTIL_MOCK_MASTERELECTION_H_
#define CERT_TRANS_UTIL_MOCK_MASTERELECTION_H_

#include <gmock/gmock.h>

#include "util/masterelection.h"

namespace cert_trans {

class MockMasterElection : public MasterElection {
 public:
  MockMasterElection() = default;

  MOCK_METHOD0(StartElection, void());
  MOCK_METHOD0(StopElection, void());
  MOCK_CONST_METHOD0(WaitToBecomeMaster, bool());
  MOCK_CONST_METHOD0(IsMaster, bool());
};

}  // namespace cert_trans


#endif  // CERT_TRANS_UTIL_MOCK_MASTERELECTION_H_
