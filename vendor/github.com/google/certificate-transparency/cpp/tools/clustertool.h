#ifndef CERT_TRANS_TOOLS_CLUSTERTOOL_H_
#define CERT_TRANS_TOOLS_CLUSTERTOOL_H_

namespace ct {
class ClusterConfig;
}  // namespace ct

namespace util {
class Status;
}  // namespace util

namespace cert_trans {


// Initialise a fresh log cluster:
//  - Creates /serving_sth containing a new STH of size zero
//  - Creates the /cluster_config entry.
template <class Logged>
util::Status InitLog(const ct::ClusterConfig& cluster_config,
                     TreeSigner<Logged>* tree_signer,
                     ConsistentStore<Logged>* consistent_store);

// Sets the cluster config
template <class Logged>
util::Status SetClusterConfig(const ct::ClusterConfig& cluster_config,
                              ConsistentStore<Logged>* consistent_store);


}  // namespace cert_trans


#endif  // CERT_TRANS_TOOLS_CLUSTERTOOL_H_
