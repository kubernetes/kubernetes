#ifndef CERT_TRANS_FETCHER_CONTINUOUS_FETCHER_H_
#define CERT_TRANS_FETCHER_CONTINUOUS_FETCHER_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "base/macros.h"
#include "fetcher/peer.h"
#include "log/database.h"
#include "log/logged_entry.h"
#include "util/executor.h"
#include "util/libevent_wrapper.h"
#include "util/task.h"

class LogVerifier;

namespace cert_trans {


class ContinuousFetcher {
 public:
  static std::unique_ptr<ContinuousFetcher> New(
      libevent::Base* base, util::Executor* executor, Database* db,
      const LogVerifier* log_verifier, bool fetch_scts);

  virtual ~ContinuousFetcher() = default;

  virtual void AddPeer(const std::string& node_id,
                       const std::shared_ptr<Peer>& peer) = 0;

  virtual void RemovePeer(const std::string& node_id) = 0;

 protected:
  ContinuousFetcher() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(ContinuousFetcher);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_FETCHER_CONTINUOUS_FETCHER_H_
