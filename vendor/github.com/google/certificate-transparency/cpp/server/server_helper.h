#ifndef CERT_TRANS_SERVER_SERVER_HELPER_H_
#define CERT_TRANS_SERVER_SERVER_HELPER_H_

#include <gflags/gflags.h>
#include <openssl/crypto.h>
#include <chrono>
#include <csignal>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>

#include "base/macros.h"
#include "log/database.h"
#include "log/file_db.h"
#include "log/file_storage.h"
#include "log/leveldb_db.h"
#include "log/sqlite_db.h"
#include "util/etcd.h"
#include "util/executor.h"
#include "util/libevent_wrapper.h"
#include "util/thread_pool.h"

namespace cert_trans {

// This includes code common to multiple CT servers. It handles parsing
// flags and creating objects that are used by multiple servers. Anything that
// is specific one type of CT server should not be in this class.
//
// Do not link server_helper into servers that don't use it as it will confuse
// the user with extra flags.
//
// Note methods named ProvideX create a new instance of X each call. Typically
// they are called once during server initialization and the return object
// lifetime is the same as that of the server.

// Calling this will CHECK if the flag validators failed to register
void EnsureValidatorsRegistered();

// Tests if the server was configured in standalone mode. Note can CHECK if
// the command line options are inconsistent. If warn_data_loss is true
// the user must set the --i_know_stand_alone_mode_can_lose_data flag as
// standalone servers are inherently prone to data loss, though useful for
// development and testing.
bool IsStandalone(bool warn_data_loss);

// Create one of the supported database types based on flags settings
std::unique_ptr<Database> ProvideDatabase();

// Create an EtcdClient implementation, either fake or real based on flags
std::unique_ptr<EtcdClient> ProvideEtcdClient(libevent::Base* event_base,
                                              ThreadPool* pool,
                                              UrlFetcher* fetcher);

}  // namespace cert_trans

#endif  // CERT_TRANS_SERVER_SERVER_HELPER_H_
