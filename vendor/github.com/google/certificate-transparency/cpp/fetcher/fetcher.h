#ifndef CERT_TRANS_FETCHER_FETCHER_H_
#define CERT_TRANS_FETCHER_FETCHER_H_

#include <memory>

#include "fetcher/peer_group.h"
#include "log/database.h"
#include "util/task.h"

class LogVerifier;

namespace cert_trans {


void FetchLogEntries(Database* db, std::unique_ptr<PeerGroup> peer_group,
                     const LogVerifier* log_verifier, util::Task* task);


}  // namespace cert_trans

#endif  // CERT_TRANS_FETCHER_FETCHER_H_
