#ifndef CERT_TRANS_UTIL_ETCD_DELETE_H_
#define CERT_TRANS_UTIL_ETCD_DELETE_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "util/etcd.h"

namespace cert_trans {


// Force delete keys in batches (implemented using concurrent
// requests). The "keys" argument are pairs of key and modified index.
void EtcdForceDeleteKeys(EtcdClient* client, std::vector<std::string>&& keys,
                         util::Task* task);


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_ETCD_DELETE_H_
