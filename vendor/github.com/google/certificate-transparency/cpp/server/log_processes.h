#ifndef CERT_TRANS_SERVER_LOG_PROCESSES_H_
#define CERT_TRANS_SERVER_LOG_PROCESSES_H_

#include <functional>

#include "log/logged_entry.h"
#include "log/tree_signer.h"

namespace cert_trans {

// Common processes that are called by server threads at intervals. This
// code is shared by binaries that write to logs, currently ct-server
// (all versions) and xjson-server.

void CleanUpEntries(ConsistentStore<LoggedEntry>* store,
                    const std::function<bool()>& is_master);

void SequenceEntries(TreeSigner<LoggedEntry>* tree_signer,
                     const std::function<bool()>& is_master);

void SignMerkleTree(TreeSigner<LoggedEntry>* tree_signer,
                    ConsistentStore<LoggedEntry>* store,
                    ClusterStateController<LoggedEntry>* controller);
}

#endif  // CERT_TRANS_SERVER_LOG_PROCESSES_H_
