#ifndef CERT_TRANS_TOOLS_CLUSTERTOOL_INL_H_
#define CERT_TRANS_TOOLS_CLUSTERTOOL_INL_H_
#include <event2/thread.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <openssl/err.h>
#include <memory>
#include <string>

#include "log/etcd_consistent_store.h"
#include "log/log_signer.h"
#include "log/logged_entry.h"
#include "log/sqlite_db.h"
#include "log/strict_consistent_store.h"
#include "log/tree_signer.h"
#include "proto/ct.pb.h"
#include "tools/clustertool.h"
#include "util/etcd.h"
#include "util/masterelection.h"
#include "util/read_key.h"
#include "util/status.h"
#include "util/thread_pool.h"


namespace cert_trans {


template <class Logged>
util::Status InitLog(const ct::ClusterConfig& cluster_config,
                     TreeSigner<Logged>* tree_signer,
                     ConsistentStore<Logged>* consistent_store) {
  if (tree_signer->UpdateTree() != TreeSigner<LoggedEntry>::OK) {
    return util::Status(util::error::UNKNOWN, "Failed to Update Tree");
  }

  util::Status status(
      consistent_store->SetServingSTH(tree_signer->LatestSTH()));
  if (!status.ok()) {
    return status;
  }

  return SetClusterConfig(cluster_config, consistent_store);
}


template <class Logged>
util::Status SetClusterConfig(const ct::ClusterConfig& cluster_config,
                              ConsistentStore<Logged>* consistent_store) {
  if (cluster_config.etcd_reject_add_pending_threshold() < 0) {
    return util::Status(util::error::INVALID_ARGUMENT,
                        "etcd_reject_add_pending_threshold cannot be less "
                        "than 0");
  }
  return consistent_store->SetClusterConfig(cluster_config);
}


}  // namespace cert_trans
#endif  // CERT_TRANS_TOOLS_CLUSTERTOOL_INL_H_
