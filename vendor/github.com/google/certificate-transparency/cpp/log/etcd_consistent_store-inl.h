#ifndef CERT_TRANS_LOG_ETCD_CONSISTENT_STORE_INL_H_
#define CERT_TRANS_LOG_ETCD_CONSISTENT_STORE_INL_H_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <unordered_map>
#include <vector>

#include "base/notification.h"
#include "log/etcd_consistent_store.h"
#include "monitoring/event_metric.h"
#include "monitoring/latency.h"
#include "monitoring/monitoring.h"
#include "util/etcd_delete.h"
#include "util/executor.h"
#include "util/masterelection.h"
#include "util/util.h"

DECLARE_int32(etcd_stats_collection_interval_seconds);

DECLARE_int32(node_state_ttl_seconds);

namespace cert_trans {
namespace {

// etcd path constants.
const char kClusterConfigFile[] = "/cluster_config";
const char kEntriesDir[] = "/entries/";
const char kSequenceFile[] = "/sequence_mapping";
const char kServingSthFile[] = "/serving_sth";
const char kNodesDir[] = "/nodes/";


static Gauge<std::string>* etcd_total_entries =
    Gauge<std::string>::New("etcd_total_entries", "type",
                            "Total number of entries in etcd by type.");

static Gauge<std::string>* etcd_store_stats =
    Gauge<std::string>::New("etcd_store_stats", "name",
                            "Re-export of etcd's store stats.");

static EventMetric<std::string> etcd_throttle_delay_ms(
    "etcd_throttle_delay_ms", "type",
    "Count and total thottle delay applied to requests, broken down by "
    "request type");

static Counter<std::string>* etcd_rejected_requests =
    Counter<std::string>::New("etcd_rejected_requests", "type",
                              "Total number of requests rejected due to "
                              "overload, broken down by request type.");

static Latency<std::chrono::milliseconds, std::string> etcd_latency_by_op_ms(
    "etcd_latency_by_op_ms", "operation",
    "Etcd latency in ms broken down by operation.");


void CheckMappingIsOrdered(const ct::SequenceMapping& mapping) {
  if (mapping.mapping_size() < 2) {
    return;
  }
  for (int64_t i = 0; i < mapping.mapping_size() - 1; ++i) {
    CHECK_LT(mapping.mapping(i).sequence_number(),
             mapping.mapping(i + 1).sequence_number());
  }
}


util::StatusOr<int64_t> GetStat(const std::map<std::string, int64_t>& stats,
                                const std::string& name) {
  const auto& it(stats.find(name));
  if (it == stats.end()) {
    return util::Status(util::error::FAILED_PRECONDITION, name + " missing.");
  }
  return it->second;
}


util::StatusOr<int64_t> CalculateNumEtcdEntries(
    const std::map<std::string, int64_t>& stats) {
  util::StatusOr<int64_t> created(GetStat(stats, "createSuccess"));
  if (!created.ok()) {
    return created;
  }

  util::StatusOr<int64_t> deleted(GetStat(stats, "deleteSuccess"));
  if (!deleted.ok()) {
    return deleted;
  }

  util::StatusOr<int64_t> compareDeleted(
      GetStat(stats, "compareAndDeleteSuccess"));
  if (!compareDeleted.ok()) {
    return compareDeleted;
  }
  util::StatusOr<int64_t> expired(GetStat(stats, "expireCount"));
  if (!expired.ok()) {
    return expired;
  }

  const int64_t num_removed(deleted.ValueOrDie() +
                            compareDeleted.ValueOrDie() +
                            expired.ValueOrDie());
  return created.ValueOrDie() - num_removed;
}

}  // namespace


template <class Logged>
EtcdConsistentStore<Logged>::EtcdConsistentStore(
    libevent::Base* base, util::Executor* executor, EtcdClient* client,
    const MasterElection* election, const std::string& root,
    const std::string& node_id)
    : client_(CHECK_NOTNULL(client)),
      base_(CHECK_NOTNULL(base)),
      executor_(CHECK_NOTNULL(executor)),
      election_(CHECK_NOTNULL(election)),
      root_(root),
      node_id_(node_id),
      serving_sth_watch_task_(CHECK_NOTNULL(executor)),
      cluster_config_watch_task_(CHECK_NOTNULL(executor)),
      etcd_stats_task_(executor_),
      received_initial_sth_(false),
      exiting_(false),
      num_etcd_entries_(0) {
  // Set up watches on things we're interested in...
  WatchServingSTH(
      std::bind(&EtcdConsistentStore<Logged>::OnEtcdServingSTHUpdated, this,
                std::placeholders::_1),
      serving_sth_watch_task_.task());
  WatchClusterConfig(
      std::bind(&EtcdConsistentStore<Logged>::OnClusterConfigUpdated, this,
                std::placeholders::_1),
      cluster_config_watch_task_.task());

  StartEtcdStatsFetch();

  // And wait for the initial updates to come back so that we've got a
  // view on the current state before proceding...
  {
    std::unique_lock<std::mutex> lock(mutex_);
    serving_sth_cv_.wait(lock, [this]() { return received_initial_sth_; });
  }
}


template <class Logged>
EtcdConsistentStore<Logged>::~EtcdConsistentStore() {
  VLOG(1) << "Cancelling watch tasks.";
  serving_sth_watch_task_.Cancel();
  cluster_config_watch_task_.Cancel();
  VLOG(1) << "Waiting for watch tasks to return.";
  serving_sth_watch_task_.Wait();
  cluster_config_watch_task_.Wait();
  VLOG(1) << "Cancelling stats task.";
  etcd_stats_task_.Cancel();
  etcd_stats_task_.Wait();
  VLOG(1) << "Joining cleanup thread";
  {
    std::lock_guard<std::mutex> lock(mutex_);
    exiting_ = true;
  }
  serving_sth_cv_.notify_all();
}


template <class Logged>
util::StatusOr<int64_t>
EtcdConsistentStore<Logged>::NextAvailableSequenceNumber() const {
  ScopedLatency scoped_latency(etcd_latency_by_op_ms.GetScopedLatency(
      "next_available_sequence_number"));

  EntryHandle<ct::SequenceMapping> sequence_mapping;
  util::Status status(GetSequenceMapping(&sequence_mapping));
  if (!status.ok()) {
    return status;
  }
  etcd_total_entries->Set("sequenced",
                          sequence_mapping.Entry().mapping_size());
  if (sequence_mapping.Entry().mapping_size() > 0) {
    return sequence_mapping.Entry()
               .mapping(sequence_mapping.Entry().mapping_size() - 1)
               .sequence_number() +
           1;
  }

  if (!serving_sth_) {
    LOG(WARNING) << "Log has no Serving STH [new log?], returning 0";
    return 0;
  }

  return serving_sth_->Entry().tree_size();
}


template <class Logged>
void EtcdConsistentStore<Logged>::WaitForServingSTHVersion(
    std::unique_lock<std::mutex>* lock, const int version) {
  VLOG(1) << "Waiting for ServingSTH version " << version;
  serving_sth_cv_.wait(*lock, [this, version]() {
    VLOG(1) << "Want version " << version << ", have: "
            << (serving_sth_ ? std::to_string(serving_sth_->Handle())
                             : "none");
    return serving_sth_ && serving_sth_->Handle() >= version;
  });
}


template <class Logged>
util::Status EtcdConsistentStore<Logged>::SetServingSTH(
    const ct::SignedTreeHead& new_sth) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("set_serving_sth"));

  const std::string full_path(GetFullPath(kServingSthFile));
  std::unique_lock<std::mutex> lock(mutex_);

  // The watcher should have already populated serving_sth_ if etcd had one.
  if (!serving_sth_) {
    // Looks like we're creating the first ever serving_sth!
    LOG(WARNING) << "Creating new " << full_path;
    // There's no current serving STH, so we can try to create one.
    EntryHandle<ct::SignedTreeHead> sth_handle(full_path, new_sth);
    util::Status status(CreateEntry(&sth_handle));
    if (!status.ok()) {
      return status;
    }
    WaitForServingSTHVersion(&lock, sth_handle.Handle());
    return util::Status::OK;
  }

  // Looks like we're updating an existing serving_sth.
  // First check that we're not trying to overwrite it with itself or an older
  // one:
  if (serving_sth_->Entry().timestamp() >= new_sth.timestamp()) {
    return util::Status(util::error::OUT_OF_RANGE,
                        "Tree head is not newer than existing head");
  }

  // Ensure that nothing weird is going on with the tree size:
  CHECK_LE(serving_sth_->Entry().tree_size(), new_sth.tree_size());

  VLOG(1) << "Updating existing " << full_path;
  EntryHandle<ct::SignedTreeHead> sth_to_etcd(full_path, new_sth,
                                              serving_sth_->Handle());
  util::Status status(UpdateEntry(&sth_to_etcd));
  if (!status.ok()) {
    return status;
  }
  WaitForServingSTHVersion(&lock, sth_to_etcd.Handle());
  return util::Status::OK;
}


template <class Logged>
util::StatusOr<ct::SignedTreeHead> EtcdConsistentStore<Logged>::GetServingSTH()
    const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (serving_sth_) {
    return serving_sth_->Entry();
  } else {
    return util::Status(util::error::NOT_FOUND, "No current Serving STH.");
  }
}


template <class Logged>
bool LeafEntriesMatch(const Logged& a, const Logged& b) {
  CHECK_EQ(a.entry().type(), b.entry().type());
  switch (a.entry().type()) {
    case ct::X509_ENTRY:
      return a.entry().x509_entry().leaf_certificate() ==
             b.entry().x509_entry().leaf_certificate();
    case ct::PRECERT_ENTRY:
      return a.entry().precert_entry().pre_certificate() ==
             b.entry().precert_entry().pre_certificate();
    case ct::PRECERT_ENTRY_V2:
      // TODO(mhs): V2 implementation required here
      LOG(FATAL) << "CT V2 not yet implemented";
      break;
    case ct::X_JSON_ENTRY:
      return a.entry().x_json_entry().json() ==
             b.entry().x_json_entry().json();
    case ct::UNKNOWN_ENTRY_TYPE:
      // Handle it below.
      break;
  }
  LOG(FATAL) << "Encountered UNKNOWN_ENTRY_TYPE:\n" << a.entry().DebugString();
}


template <class Logged>
util::Status EtcdConsistentStore<Logged>::AddPendingEntry(Logged* entry) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("add_pending_entry"));

  CHECK_NOTNULL(entry);
  CHECK(!entry->has_sequence_number());

  util::Status status(MaybeReject("add_pending_entry"));
  if (!status.ok()) {
    return status;
  }

  const std::string full_path(GetEntryPath(*entry));
  EntryHandle<Logged> handle(full_path, *entry);
  status = CreateEntry(&handle);
  if (status.CanonicalCode() == util::error::FAILED_PRECONDITION) {
    // Entry with that hash already exists.
    EntryHandle<Logged> preexisting_entry;
    status = GetEntry(full_path, &preexisting_entry);
    if (!status.ok()) {
      LOG(ERROR) << "Couldn't create or fetch " << full_path << " : "
                 << status;
      return status;
    }

    // Check the leaf certs are the same (we might be seeing the same cert
    // submitted with a different chain.)
    CHECK(LeafEntriesMatch(preexisting_entry.Entry(), *entry));
    *entry->mutable_sct() = preexisting_entry.Entry().sct();
    return util::Status(util::error::ALREADY_EXISTS,
                        "Pending entry already exists.");
  }
  return status;
}

template <class Logged>
util::Status EtcdConsistentStore<Logged>::GetPendingEntryForHash(
    const std::string& hash, EntryHandle<Logged>* entry) const {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("get_pending_entry_for_hash"));

  util::Status status(GetEntry(GetEntryPath(hash), entry));
  if (status.ok()) {
    CHECK(!entry->Entry().has_sequence_number());
  }

  return status;
}


template <class Logged>
util::Status EtcdConsistentStore<Logged>::GetPendingEntries(
    std::vector<EntryHandle<Logged>>* entries) const {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("get_pending_entries"));

  util::Status status(GetAllEntriesInDir(GetFullPath(kEntriesDir), entries));
  if (status.ok()) {
    for (const auto& entry : *entries) {
      CHECK(!entry.Entry().has_sequence_number());
    }
  }
  etcd_total_entries->Set("entries", entries->size());
  return status;
}


template <class Logged>
util::Status EtcdConsistentStore<Logged>::GetSequenceMapping(
    EntryHandle<ct::SequenceMapping>* sequence_mapping) const {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("get_sequence_mapping"));

  util::Status status(GetEntry(GetFullPath(kSequenceFile), sequence_mapping));
  if (!status.ok()) {
    return status;
  }
  CheckMappingIsOrdered(sequence_mapping->Entry());
  CheckMappingIsContiguousWithServingTree(sequence_mapping->Entry());
  etcd_total_entries->Set("sequenced",
                          sequence_mapping->Entry().mapping_size());
  return util::Status::OK;
}


template <class Logged>
util::Status EtcdConsistentStore<Logged>::UpdateSequenceMapping(
    EntryHandle<ct::SequenceMapping>* entry) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("update_sequence_mapping"));

  CHECK(entry->HasHandle());
  CheckMappingIsOrdered(entry->Entry());
  CheckMappingIsContiguousWithServingTree(entry->Entry());
  return UpdateEntry(entry);
}


template <class Logged>
util::StatusOr<ct::ClusterNodeState>
EtcdConsistentStore<Logged>::GetClusterNodeState() const {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("get_cluster_node_state"));

  EntryHandle<ct::ClusterNodeState> handle;
  util::Status status(GetEntry(GetNodePath(node_id_), &handle));
  if (!status.ok()) {
    return status;
  }
  return handle.Entry();
}


template <class Logged>
util::Status EtcdConsistentStore<Logged>::SetClusterNodeState(
    const ct::ClusterNodeState& state) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("set_cluster_node_state"));

  // TODO(alcutter): consider keeping the handle for this around to check that
  // nobody else is updating our cluster state.
  ct::ClusterNodeState local_state(state);
  local_state.set_node_id(node_id_);
  EntryHandle<ct::ClusterNodeState> entry(GetNodePath(node_id_), local_state);
  const std::chrono::seconds ttl(FLAGS_node_state_ttl_seconds);
  return ForceSetEntryWithTTL(ttl, &entry);
}


// static
template <class Logged>
template <class T, class CB>
void EtcdConsistentStore<Logged>::ConvertSingleUpdate(
    const std::string& full_path, const CB& callback,
    const std::vector<EtcdClient::Node>& updates) {
  CHECK_LE(static_cast<size_t>(0), updates.size());
  if (updates.empty()) {
    EntryHandle<T> handle;
    handle.SetKey(full_path);
    callback(Update<T>(handle, false /* exists */));
  } else {
    callback(TypedUpdateFromNode<T>(updates[0]));
  }
}


// static
template <class Logged>
template <class T, class CB>
void EtcdConsistentStore<Logged>::ConvertMultipleUpdate(
    const CB& callback, const std::vector<EtcdClient::Node>& watch_updates) {
  std::vector<Update<T>> updates;
  for (auto& w : watch_updates) {
    updates.emplace_back(TypedUpdateFromNode<T>(w));
  }
  callback(updates);
}


template <class Logged>
void EtcdConsistentStore<Logged>::WatchServingSTH(
    const typename ConsistentStore<Logged>::ServingSTHCallback& cb,
    util::Task* task) {
  const std::string full_path(GetFullPath(kServingSthFile));
  client_->Watch(
      full_path,
      std::bind(&ConvertSingleUpdate<
                    ct::SignedTreeHead,
                    typename ConsistentStore<Logged>::ServingSTHCallback>,
                full_path, cb, std::placeholders::_1),
      task);
}


template <class Logged>
void EtcdConsistentStore<Logged>::WatchClusterNodeStates(
    const typename ConsistentStore<Logged>::ClusterNodeStateCallback& cb,
    util::Task* task) {
  client_->Watch(
      GetFullPath(kNodesDir),
      std::bind(
          &ConvertMultipleUpdate<
              ct::ClusterNodeState,
              typename ConsistentStore<Logged>::ClusterNodeStateCallback>,
          cb, std::placeholders::_1),
      task);
}


template <class Logged>
void EtcdConsistentStore<Logged>::WatchClusterConfig(
    const typename ConsistentStore<Logged>::ClusterConfigCallback& cb,
    util::Task* task) {
  const std::string full_path(GetFullPath(kClusterConfigFile));
  client_->Watch(
      full_path,
      std::bind(&ConvertSingleUpdate<
                    ct::ClusterConfig,
                    typename ConsistentStore<Logged>::ClusterConfigCallback>,
                full_path, cb, std::placeholders::_1),
      task);
}


template <class Logged>
util::Status EtcdConsistentStore<Logged>::SetClusterConfig(
    const ct::ClusterConfig& config) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("set_cluster_config"));

  EntryHandle<ct::ClusterConfig> entry(GetFullPath(kClusterConfigFile),
                                       config);
  return ForceSetEntry(&entry);
}


template <class Logged>
template <class T>
util::Status EtcdConsistentStore<Logged>::GetEntry(
    const std::string& path, EntryHandle<T>* entry) const {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("get_entry"));

  CHECK_NOTNULL(entry);
  util::SyncTask task(executor_);
  EtcdClient::GetResponse resp;
  client_->Get(path, &resp, task.task());
  task.Wait();
  if (!task.status().ok()) {
    return task.status();
  }
  T t;
  CHECK(t.ParseFromString(util::FromBase64(resp.node.value_.c_str())));
  entry->Set(path, t, resp.node.modified_index_);
  return util::Status::OK;
}


template <class Logged>
template <class T>
util::Status EtcdConsistentStore<Logged>::GetAllEntriesInDir(
    const std::string& dir, std::vector<EntryHandle<T>>* entries) const {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("get_all_entries_in_dir"));

  CHECK_NOTNULL(entries);
  CHECK_EQ(static_cast<size_t>(0), entries->size());
  util::SyncTask task(executor_);
  EtcdClient::GetResponse resp;
  client_->Get(dir, &resp, task.task());
  task.Wait();
  if (!task.status().ok()) {
    return task.status();
  }
  if (!resp.node.is_dir_) {
    return util::Status(util::error::FAILED_PRECONDITION,
                        "node is not a directory: " + dir);
  }
  for (const auto& node : resp.node.nodes_) {
    T t;
    CHECK(t.ParseFromString(util::FromBase64(node.value_.c_str())));
    entries->emplace_back(
        EntryHandle<Logged>(node.key_, t, node.modified_index_));
  }
  return util::Status::OK;
}


template <class Logged>
template <class T>
util::Status EtcdConsistentStore<Logged>::UpdateEntry(EntryHandle<T>* t) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("update_entry"));

  CHECK_NOTNULL(t);
  CHECK(t->HasHandle());
  CHECK(t->HasKey());
  std::string flat_entry;
  CHECK(t->Entry().SerializeToString(&flat_entry));
  util::SyncTask task(executor_);
  EtcdClient::Response resp;
  client_->Update(t->Key(), util::ToBase64(flat_entry), t->Handle(), &resp,
                  task.task());
  task.Wait();
  if (task.status().ok()) {
    t->SetHandle(resp.etcd_index);
  }
  return task.status();
}


template <class Logged>
template <class T>
util::Status EtcdConsistentStore<Logged>::CreateEntry(EntryHandle<T>* t) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("create_entry"));

  CHECK_NOTNULL(t);
  CHECK(!t->HasHandle());
  CHECK(t->HasKey());
  std::string flat_entry;
  CHECK(t->Entry().SerializeToString(&flat_entry));
  util::SyncTask task(executor_);
  EtcdClient::Response resp;
  client_->Create(t->Key(), util::ToBase64(flat_entry), &resp, task.task());
  task.Wait();
  if (task.status().ok()) {
    t->SetHandle(resp.etcd_index);
  }
  return task.status();
}


template <class Logged>
template <class T>
util::Status EtcdConsistentStore<Logged>::ForceSetEntry(EntryHandle<T>* t) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("force_set_entry"));

  CHECK_NOTNULL(t);
  CHECK(t->HasKey());
  // For now we check that |t| wasn't fetched from the etcd store (i.e. it's a
  // new EntryHandle.  The reason is that if it had been fetched, then the
  // calling code should be doing an UpdateEntry() here since they have the
  // handle.
  CHECK(!t->HasHandle());
  std::string flat_entry;
  CHECK(t->Entry().SerializeToString(&flat_entry));
  util::SyncTask task(executor_);
  EtcdClient::Response resp;
  client_->ForceSet(t->Key(), util::ToBase64(flat_entry), &resp, task.task());
  task.Wait();
  if (task.status().ok()) {
    t->SetHandle(resp.etcd_index);
  }
  return task.status();
}


template <class Logged>
template <class T>
util::Status EtcdConsistentStore<Logged>::ForceSetEntryWithTTL(
    const std::chrono::seconds& ttl, EntryHandle<T>* t) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("force_set_entry_with_ttl"));

  CHECK_NOTNULL(t);
  CHECK(t->HasKey());
  // For now we check that |t| wasn't fetched from the etcd store (i.e. it's a
  // new EntryHandle.  The reason is that if it had been fetched, then the
  // calling code should be doing an UpdateEntryWithTTL() here since they have
  // the handle.
  CHECK(!t->HasHandle());
  CHECK_LE(0, ttl.count());
  std::string flat_entry;
  CHECK(t->Entry().SerializeToString(&flat_entry));
  util::SyncTask task(executor_);
  EtcdClient::Response resp;
  client_->ForceSetWithTTL(t->Key(), util::ToBase64(flat_entry), ttl, &resp,
                           task.task());
  task.Wait();
  if (task.status().ok()) {
    t->SetHandle(resp.etcd_index);
  }
  return task.status();
}


template <class Logged>
template <class T>
util::Status EtcdConsistentStore<Logged>::DeleteEntry(EntryHandle<T>* entry) {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("delete_entry"));

  CHECK_NOTNULL(entry);
  CHECK(entry->HasHandle());
  CHECK(entry->HasKey());
  util::SyncTask task(executor_);
  client_->Delete(entry->Key(), entry->Handle(), task.task());
  task.Wait();
  return task.status();
}


template <class Logged>
std::string EtcdConsistentStore<Logged>::GetEntryPath(
    const Logged& entry) const {
  return GetEntryPath(entry.Hash());
}


template <class Logged>
std::string EtcdConsistentStore<Logged>::GetEntryPath(
    const std::string& hash) const {
  return GetFullPath(std::string(kEntriesDir) + util::HexString(hash));
}


template <class Logged>
std::string EtcdConsistentStore<Logged>::GetNodePath(
    const std::string& id) const {
  return GetFullPath(std::string(kNodesDir) + id);
}


template <class Logged>
std::string EtcdConsistentStore<Logged>::GetFullPath(
    const std::string& key) const {
  CHECK(key.size() > 0);
  CHECK_EQ('/', key[0]);
  return root_ + key;
}


template <class Logged>
void EtcdConsistentStore<Logged>::CheckMappingIsContiguousWithServingTree(
    const ct::SequenceMapping& mapping) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (serving_sth_ && mapping.mapping_size() > 0) {
    // The sequence numbers are signed. However the tree size must fit in
    // memory so the unsigned -> signed conversion below should not overflow.
    CHECK_LE(serving_sth_->Entry().tree_size(), INT64_MAX);

    const int64_t tree_size(serving_sth_->Entry().tree_size());
    // The mapping must not have a gap between its lowest mapping and the
    // serving tree
    const int64_t lowest_sequence_number(mapping.mapping(0).sequence_number());
    CHECK_LE(lowest_sequence_number, tree_size);
    // It must also be contiguous for all entries not yet included in the
    // serving tree. (Note that entries below that may not be contiguous
    // because the clean-up operation may not remove them in order.)
    bool above_sth(false);
    for (int i(0); i < mapping.mapping_size() - 1; ++i) {
      const int64_t mapped_seq(mapping.mapping(i).sequence_number());
      if (mapped_seq >= tree_size) {
        CHECK_EQ(mapped_seq + 1, mapping.mapping(i + 1).sequence_number());
        above_sth = true;
      } else {
        CHECK(!above_sth);
      }
    }
  }
}


// static
template <class Logged>
template <class T>
Update<T> EtcdConsistentStore<Logged>::TypedUpdateFromNode(
    const EtcdClient::Node& node) {
  const std::string raw_value(util::FromBase64(node.value_.c_str()));
  T thing;
  CHECK(thing.ParseFromString(raw_value)) << raw_value;
  EntryHandle<T> handle(node.key_, thing);
  if (!node.deleted_) {
    handle.SetHandle(node.modified_index_);
  }
  return Update<T>(handle, !node.deleted_);
}


template <class Logged>
void EtcdConsistentStore<Logged>::UpdateLocalServingSTH(
    const std::unique_lock<std::mutex>& lock,
    const EntryHandle<ct::SignedTreeHead>& handle) {
  CHECK(lock.owns_lock());
  CHECK(!serving_sth_ ||
        serving_sth_->Entry().timestamp() < handle.Entry().timestamp());

  VLOG(1) << "Updating serving_sth_ to: " << handle.Entry().DebugString();
  serving_sth_.reset(new EntryHandle<ct::SignedTreeHead>(handle));
}


template <class Logged>
void EtcdConsistentStore<Logged>::OnEtcdServingSTHUpdated(
    const Update<ct::SignedTreeHead>& update) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (update.exists_) {
    VLOG(1) << "Got ServingSTH version " << update.handle_.Handle() << ": "
            << update.handle_.Entry().DebugString();
    UpdateLocalServingSTH(lock, update.handle_);
  } else {
    LOG(WARNING) << "ServingSTH non-existent/deleted.";
    // TODO(alcutter): What to do here?
    serving_sth_.reset();
  }
  received_initial_sth_ = true;
  lock.unlock();
  serving_sth_cv_.notify_all();
}


template <class Logged>
void EtcdConsistentStore<Logged>::OnClusterConfigUpdated(
    const Update<ct::ClusterConfig>& update) {
  if (update.exists_) {
    VLOG(1) << "Got ClusterConfig version " << update.handle_.Handle() << ": "
            << update.handle_.Entry().DebugString();
    std::lock_guard<std::mutex> lock(mutex_);
    cluster_config_.reset(new ct::ClusterConfig(update.handle_.Entry()));
  } else {
    LOG(WARNING) << "ClusterConfig non-existent/deleted.";
    // TODO(alcutter): What to do here?
  }
}


template <class Logged>
util::StatusOr<int64_t> EtcdConsistentStore<Logged>::CleanupOldEntries() {
  ScopedLatency scoped_latency(
      etcd_latency_by_op_ms.GetScopedLatency("cleanup_old_entries"));

  if (!election_->IsMaster()) {
    return util::Status(util::error::PERMISSION_DENIED,
                        "Non-master node cannot run cleanups.");
  }

  // Figure out where we're cleaning up to...
  std::unique_lock<std::mutex> lock(mutex_);
  if (!serving_sth_) {
    LOG(INFO) << "No current serving_sth, nothing to do.";
    return 0;
  }
  const int64_t clean_up_to_sequence_number(serving_sth_->Entry().tree_size() -
                                            1);
  lock.unlock();

  LOG(INFO) << "Cleaning old entries up to and including sequence number: "
            << clean_up_to_sequence_number;

  EntryHandle<ct::SequenceMapping> sequence_mapping;
  util::Status status(GetSequenceMapping(&sequence_mapping));
  if (!status.ok()) {
    LOG(WARNING) << "Couldn't get sequence mapping: " << status;
    return status;
  }

  std::vector<std::string> keys_to_delete;
  for (int mapping_index = 0;
       mapping_index < sequence_mapping.Entry().mapping_size() &&
       sequence_mapping.Entry().mapping(mapping_index).sequence_number() <=
           clean_up_to_sequence_number;
       ++mapping_index) {
    // Delete the entry from /entries.
    keys_to_delete.emplace_back(GetEntryPath(
        sequence_mapping.Entry().mapping(mapping_index).entry_hash()));
  }


  const int64_t num_entries_cleaned(keys_to_delete.size());
  util::SyncTask task(executor_);
  EtcdForceDeleteKeys(client_, std::move(keys_to_delete), task.task());
  task.Wait();
  status = task.status();
  if (!status.ok()) {
    LOG(WARNING) << "EtcdDeleteKeys failed: " << task.status();
  }
  return num_entries_cleaned;
}


template <class Logged>
void EtcdConsistentStore<Logged>::StartEtcdStatsFetch() {
  if (etcd_stats_task_.task()->CancelRequested()) {
    etcd_stats_task_.task()->Return(util::Status::CANCELLED);
    return;
  }
  EtcdClient::StatsResponse* response(new EtcdClient::StatsResponse);
  util::Task* stats_task(etcd_stats_task_.task()->AddChild(
      std::bind(&EtcdConsistentStore<Logged>::EtcdStatsFetchDone, this,
                response, std::placeholders::_1)));
  client_->GetStoreStats(response, stats_task);
}


template <class Logged>
void EtcdConsistentStore<Logged>::EtcdStatsFetchDone(
    EtcdClient::StatsResponse* response, util::Task* task) {
  CHECK_NOTNULL(response);
  CHECK_NOTNULL(task);
  std::unique_ptr<EtcdClient::StatsResponse> response_deleter(response);
  if (task->status().ok()) {
    for (const auto& stat : response->stats) {
      VLOG(2) << "etcd stat: " << stat.first << " = " << stat.second;
      etcd_store_stats->Set(stat.first, stat.second);
    }
    const util::StatusOr<int64_t> num_entries(
        CalculateNumEtcdEntries(response->stats));
    if (num_entries.ok()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        num_etcd_entries_ = num_entries.ValueOrDie();
      }
      etcd_total_entries->Set("all", num_etcd_entries_);
    } else {
      VLOG(1) << "Failed to calculate num_entries: " << num_entries.status();
    }
  } else {
    LOG(WARNING) << "Etcd stats fetch failed: " << task->status();
  }

  base_->Delay(
      std::chrono::seconds(FLAGS_etcd_stats_collection_interval_seconds),
      etcd_stats_task_.task()->AddChild(
          std::bind(&EtcdConsistentStore<Logged>::StartEtcdStatsFetch, this)));
}

// This method attempts to modulate the incoming traffic in response to the
// number of entries currently in etcd.
//
// Once the number of entries is above reject_threshold, we will start
// returning a RESOURCE_EXHAUSTED status, which should result in a 503 being
// sent to the client.
template <class Logged>
util::Status EtcdConsistentStore<Logged>::MaybeReject(
    const std::string& type) const {
  std::unique_lock<std::mutex> lock(mutex_);

  if (!cluster_config_) {
    // No config, whatever.
    return util::Status::OK;
  }

  const int64_t etcd_size(num_etcd_entries_);
  const int64_t reject_threshold(
      cluster_config_->etcd_reject_add_pending_threshold());
  lock.unlock();

  if (etcd_size >= reject_threshold) {
    etcd_rejected_requests->Increment(type);
    return util::Status(util::error::RESOURCE_EXHAUSTED,
                        "Rejected due to high number of pending entries.");
  }
  return util::Status::OK;
}


}  // namespace cert_trans

#endif  // CERT_TRANS_LOG_ETCD_CONSISTENT_STORE_INL_H_
