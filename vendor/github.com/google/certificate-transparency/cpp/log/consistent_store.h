#ifndef CERT_TRANS_LOG_CONSISTENT_STORE_H_
#define CERT_TRANS_LOG_CONSISTENT_STORE_H_

#include <stdint.h>
#include <mutex>
#include <vector>

#include "base/macros.h"
#include "proto/ct.pb.h"
#include "util/status.h"
#include "util/statusor.h"
#include "util/task.h"


namespace cert_trans {


template <class Logged>
class EtcdConsistentStore;


// Wraps an instance of |T| and associates it with a versioning handle
// (required for atomic 'compare-and-update' semantics.)
template <class T>
class EntryHandle {
 public:
  EntryHandle() = default;

  const T& Entry() const {
    return entry_;
  }

  T* MutableEntry() {
    return &entry_;
  }

  bool HasHandle() const {
    return has_handle_;
  }

  int Handle() const {
    return handle_;
  }

  bool HasKey() const {
    return !key_.empty();
  }

  const std::string& Key() const {
    return key_;
  }

 private:
  EntryHandle(const T& entry, int handle)
      : entry_(entry), has_handle_(true), handle_(handle) {
  }

  EntryHandle(const std::string& key, const T& entry, int handle)
      : key_(key), entry_(entry), has_handle_(true), handle_(handle) {
  }

  explicit EntryHandle(const std::string& key, const T& entry)
      : key_(key), entry_(entry), has_handle_(false) {
  }

  void Set(const std::string& key, const T& entry, int handle) {
    key_ = key;
    entry_ = entry;
    handle_ = handle;
    has_handle_ = true;
  }

  void SetHandle(int new_handle) {
    handle_ = new_handle;
    has_handle_ = true;
  }

  void SetKey(const std::string& key) {
    key_ = key;
  }

  std::string key_;
  T entry_;
  bool has_handle_;
  int handle_;

  template <class Logged>
  friend class EtcdConsistentStore;
  friend class EtcdConsistentStoreTest;
};


template <class T>
struct Update {
  Update(const EntryHandle<T>& handle, bool exists)
      : handle_(handle), exists_(exists) {
  }

  Update(const Update& other) = default;

  const EntryHandle<T> handle_;
  const bool exists_;
};


template <class Logged>
class ConsistentStore {
 public:
  typedef std::function<void(const Update<ct::SignedTreeHead>& update)>
      ServingSTHCallback;
  typedef std::function<void(const std::vector<Update<ct::ClusterNodeState>>&
                                 updates)> ClusterNodeStateCallback;
  typedef std::function<void(const Update<ct::ClusterConfig>& update)>
      ClusterConfigCallback;

  ConsistentStore() = default;

  virtual ~ConsistentStore() = default;

  virtual util::StatusOr<int64_t> NextAvailableSequenceNumber() const = 0;

  virtual util::Status SetServingSTH(const ct::SignedTreeHead& new_sth) = 0;

  virtual util::StatusOr<ct::SignedTreeHead> GetServingSTH() const = 0;

  virtual util::Status AddPendingEntry(Logged* entry) = 0;

  virtual util::Status GetPendingEntryForHash(
      const std::string& hash, EntryHandle<Logged>* entry) const = 0;

  virtual util::Status GetPendingEntries(
      std::vector<EntryHandle<Logged>>* entries) const = 0;

  virtual util::Status GetSequenceMapping(
      EntryHandle<ct::SequenceMapping>* entry) const = 0;

  virtual util::Status UpdateSequenceMapping(
      EntryHandle<ct::SequenceMapping>* entry) = 0;

  virtual util::StatusOr<ct::ClusterNodeState> GetClusterNodeState() const = 0;

  virtual util::Status SetClusterNodeState(
      const ct::ClusterNodeState& state) = 0;

  virtual void WatchServingSTH(const ServingSTHCallback& cb,
                               util::Task* task) = 0;

  virtual void WatchClusterNodeStates(const ClusterNodeStateCallback& cb,
                                      util::Task* task) = 0;

  virtual void WatchClusterConfig(const ClusterConfigCallback& cb,
                                  util::Task* task) = 0;

  virtual util::Status SetClusterConfig(const ct::ClusterConfig& config) = 0;

  // Cleans up entries in the store according to the implementation's policy.
  // Returns either the number of entries cleaned up, or a Status describing
  // the error.
  virtual util::StatusOr<int64_t> CleanupOldEntries() = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(ConsistentStore);
};


}  // namespace

#endif  // CERT_TRANS_LOG_CONSISTENT_STORE_H_
