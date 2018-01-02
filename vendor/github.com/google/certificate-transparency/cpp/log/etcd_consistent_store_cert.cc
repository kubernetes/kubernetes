#include <gflags/gflags.h>

#include "log/etcd_consistent_store-inl.h"
#include "log/logged_entry.h"

// This needs to be quite frequent since the number of entries which can be
// added every second can be pretty high.
DEFINE_int32(etcd_stats_collection_interval_seconds, 2,
             "Number of seconds between fetches of etcd stats.");
DEFINE_int32(node_state_ttl_seconds, 60,
             "TTL in seconds on the node state files.");

namespace cert_trans {
template class EtcdConsistentStore<LoggedEntry>;
}  // namespace cert_trans
