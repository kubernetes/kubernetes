#ifndef CERT_TRANS_FETCHER_MOCK_CONTINUOUS_FETCHER_H_
#define CERT_TRANS_FETCHER_MOCK_CONTINUOUS_FETCHER_H_

#include <gmock/gmock.h>

#include "fetcher/continuous_fetcher.h"

namespace cert_trans {


class MockContinuousFetcher : public ContinuousFetcher {
 public:
  MOCK_METHOD2(AddPeer, void(const std::string& node_id,
                             const std::shared_ptr<Peer>& peer));
  MOCK_METHOD1(RemovePeer, void(const std::string& node_id));
};


}  // namespace cert_trans

#endif  // CERT_TRANS_FETCHER_MOCK_CONTINUOUS_FETCHER_H_
