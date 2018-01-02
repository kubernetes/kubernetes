#include "log/file_db.h"

#include <glog/logging.h>
#include <stdint.h>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "log/file_storage.h"
#include "monitoring/latency.h"
#include "monitoring/monitoring.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/util.h"

using std::chrono::milliseconds;
using std::lock_guard;
using std::make_pair;
using std::min;
using std::mutex;
using std::set;
using std::stoll;
using std::string;
using std::to_string;
using std::unique_lock;
using std::unique_ptr;

namespace cert_trans {
namespace {


static Latency<milliseconds, string> latency_by_op_ms(
    "filedb_latency_by_operation_ms", "operation",
    "Database latency in ms broken out by operation.");


const char kMetaNodeIdKey[] = "node_id";


string FormatSequenceNumber(const int64_t seq) {
  return to_string(seq);
}


int64_t ParseSequenceNumber(const string& seq) {
  return stoll(seq);
}


}  // namespace


const size_t FileDB::kTimestampBytesIndexed = 6;


class FileDB::Iterator : public Database::Iterator {
 public:
  Iterator(const FileDB* db, int64_t start_index)
      : db_(CHECK_NOTNULL(db)), next_index_(start_index) {
    CHECK_GE(next_index_, 0);
  }

  bool GetNextEntry(LoggedEntry* entry) override {
    CHECK_NOTNULL(entry);
    {
      lock_guard<mutex> lock(db_->lock_);
      if (next_index_ >= db_->contiguous_size_) {
        set<int64_t>::const_iterator it(
            db_->sparse_entries_.lower_bound(next_index_));
        if (it == db_->sparse_entries_.end()) {
          return false;
        }

        next_index_ = *it;
      }
    }

    CHECK_EQ(db_->LookupByIndex(next_index_, entry), Database::LOOKUP_OK);
    ++next_index_;
    return true;
  }

 private:
  const FileDB* const db_;
  int64_t next_index_;
};


FileDB::FileDB(FileStorage* cert_storage, FileStorage* tree_storage,
               FileStorage* meta_storage)
    : cert_storage_(CHECK_NOTNULL(cert_storage)),
      tree_storage_(CHECK_NOTNULL(tree_storage)),
      meta_storage_(CHECK_NOTNULL(meta_storage)),
      contiguous_size_(0),
      latest_tree_timestamp_(0) {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("open"));
  BuildIndex();
}


FileDB::~FileDB() {
}


Database::WriteResult FileDB::CreateSequencedEntry_(
    const LoggedEntry& logged) {
  CHECK(logged.has_sequence_number());
  CHECK_GE(logged.sequence_number(), 0);
  ScopedLatency latency(
      latency_by_op_ms.GetScopedLatency("create_sequenced_entry"));

  string data;
  CHECK(logged.SerializeToString(&data));

  const string seq_str(FormatSequenceNumber(logged.sequence_number()));

  unique_lock<mutex> lock(lock_);

  // Try to create.
  util::Status status(cert_storage_->CreateEntry(seq_str, data));
  if (status.CanonicalCode() == util::error::ALREADY_EXISTS) {
    string existing_data;
    status = cert_storage_->LookupEntry(seq_str, &existing_data);
    CHECK_EQ(status, util::Status::OK);
    if (existing_data == data) {
      return this->OK;
    }
    return this->SEQUENCE_NUMBER_ALREADY_IN_USE;
  }
  CHECK_EQ(status, util::Status::OK);

  InsertEntryMapping(logged.sequence_number(), logged.Hash());

  return this->OK;
}


Database::LookupResult FileDB::LookupByHash(const string& hash,
                                            LoggedEntry* result) const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("lookup_by_hash"));

  unique_lock<mutex> lock(lock_);

  auto i(id_by_hash_.find(hash));
  if (i == id_by_hash_.end()) {
    return this->NOT_FOUND;
  }
  const string seq_str(FormatSequenceNumber(i->second));

  lock.unlock();

  string cert_data;
  const util::Status status(cert_storage_->LookupEntry(seq_str, &cert_data));
  // Gotta be there, or we're in trouble...
  CHECK_EQ(status, util::Status::OK);

  LoggedEntry logged;
  CHECK(logged.ParseFromString(cert_data));
  CHECK_EQ(logged.Hash(), hash);

  if (result) {
    logged.Swap(result);
  }

  return this->LOOKUP_OK;
}


Database::LookupResult FileDB::LookupByIndex(int64_t sequence_number,
                                             LoggedEntry* result) const {
  CHECK_GE(sequence_number, 0);
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("lookup_by_index"));

  const string seq_str(FormatSequenceNumber(sequence_number));
  string cert_data;
  if (cert_storage_->LookupEntry(seq_str, &cert_data).CanonicalCode() ==
      util::error::NOT_FOUND) {
    return this->NOT_FOUND;
  }
  if (result) {
    CHECK(result->ParseFromString(cert_data));
    CHECK_EQ(result->sequence_number(), sequence_number);
  }
  return this->LOOKUP_OK;
}


unique_ptr<Database::Iterator> FileDB::ScanEntries(int64_t start_index) const {
  return unique_ptr<Iterator>(new Iterator(this, start_index));
}


Database::WriteResult FileDB::WriteTreeHead_(const ct::SignedTreeHead& sth) {
  CHECK_GE(sth.tree_size(), 0);
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("write_tree_head"));

  // 6 bytes are good enough for some 9000 years.
  string timestamp_key =
      Serializer::SerializeUint(sth.timestamp(),
                                FileDB::kTimestampBytesIndexed);
  string data;
  CHECK(sth.SerializeToString(&data));

  unique_lock<mutex> lock(lock_);
  util::Status status(tree_storage_->CreateEntry(timestamp_key, data));
  if (status.CanonicalCode() == util::error::ALREADY_EXISTS) {
    string existing_sth_data;
    status = tree_storage_->LookupEntry(timestamp_key, &existing_sth_data);
    CHECK_EQ(status, util::Status::OK);
    if (existing_sth_data == data) {
      LOG(WARNING) << "Attempted to store identical STH in DB.";
      return this->OK;
    }
    return this->DUPLICATE_TREE_HEAD_TIMESTAMP;
  }
  CHECK_EQ(status, util::Status::OK);

  if (sth.timestamp() > latest_tree_timestamp_) {
    latest_tree_timestamp_ = sth.timestamp();
    latest_timestamp_key_ = timestamp_key;
  }

  lock.unlock();
  callbacks_.Call(sth);

  return this->OK;
}


Database::LookupResult FileDB::LatestTreeHead(
    ct::SignedTreeHead* result) const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("latest_tree_head"));
  lock_guard<mutex> lock(lock_);

  return LatestTreeHeadNoLock(result);
}


int64_t FileDB::TreeSize() const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("tree_size"));
  lock_guard<mutex> lock(lock_);

  return contiguous_size_;
}


void FileDB::AddNotifySTHCallback(
    const Database::NotifySTHCallback* callback) {
  unique_lock<mutex> lock(lock_);

  callbacks_.Add(callback);

  ct::SignedTreeHead sth;
  if (LatestTreeHeadNoLock(&sth) == this->LOOKUP_OK) {
    lock.unlock();
    (*callback)(sth);
  }
}


void FileDB::RemoveNotifySTHCallback(
    const Database::NotifySTHCallback* callback) {
  lock_guard<mutex> lock(lock_);

  callbacks_.Remove(callback);
}


void FileDB::InitializeNode(const string& node_id) {
  CHECK(!node_id.empty());
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("initialize_node"));
  unique_lock<mutex> lock(lock_);
  string existing_id;
  if (NodeId(&existing_id) != this->NOT_FOUND) {
    LOG(FATAL) << "Attempting to initialze DB belonging to node with node_id: "
               << existing_id;
  }
  CHECK(meta_storage_->CreateEntry(kMetaNodeIdKey, node_id).ok());
}


Database::LookupResult FileDB::NodeId(string* node_id) {
  CHECK_NOTNULL(node_id);
  if (!meta_storage_->LookupEntry(kMetaNodeIdKey, node_id).ok()) {
    return this->NOT_FOUND;
  }
  return this->LOOKUP_OK;
}


void FileDB::BuildIndex() {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("build_index"));
  // Technically, this should only be called from the constructor, so
  // this should not be necessarily, but just to be sure...
  lock_guard<mutex> lock(lock_);

  const set<string> sequence_numbers(cert_storage_->Scan());
  id_by_hash_.reserve(sequence_numbers.size());

  for (const auto& seq_path : sequence_numbers) {
    const int64_t seq(ParseSequenceNumber(seq_path));
    string cert_data;
    // Read the data; tolerate no errors.
    CHECK_EQ(cert_storage_->LookupEntry(seq_path, &cert_data),
             util::Status::OK)
        << "Failed to read entry with sequence number " << seq;

    LoggedEntry logged;
    CHECK(logged.ParseFromString(cert_data))
        << "Failed to parse entry with sequence number " << seq;
    CHECK(logged.has_sequence_number())
        << "sequence_number() is unset for for entry with sequence number "
        << seq;
    CHECK_EQ(logged.sequence_number(), seq)
        << "Entry has a negative sequence_number(): " << seq;

    InsertEntryMapping(logged.sequence_number(), logged.Hash());
  }

  // Now read the STH entries.
  set<string> sth_timestamps = tree_storage_->Scan();
  if (!sth_timestamps.empty()) {
    latest_timestamp_key_ = *sth_timestamps.rbegin();
    CHECK_EQ(DeserializeResult::OK,
             Deserializer::DeserializeUint<uint64_t>(
                 latest_timestamp_key_, FileDB::kTimestampBytesIndexed,
                 &latest_tree_timestamp_));
  }
}


Database::LookupResult FileDB::LatestTreeHeadNoLock(
    ct::SignedTreeHead* result) const {
  if (latest_tree_timestamp_ == 0) {
    return this->NOT_FOUND;
  }

  string tree_data;
  CHECK_EQ(tree_storage_->LookupEntry(latest_timestamp_key_, &tree_data),
           util::Status::OK);

  CHECK(result->ParseFromString(tree_data));
  CHECK_EQ(result->timestamp(), latest_tree_timestamp_);

  return this->LOOKUP_OK;
}


// This must be called with "lock_" held.
void FileDB::InsertEntryMapping(int64_t sequence_number, const string& hash) {
  if (!id_by_hash_.insert(make_pair(hash, sequence_number)).second) {
    // This is a duplicate hash under a new sequence number.
    // Make sure we track the entry with the lowest sequence number:
    id_by_hash_[hash] = min(id_by_hash_[hash], sequence_number);
  }

  if (sequence_number == contiguous_size_) {
    ++contiguous_size_;
    for (auto i = sparse_entries_.find(contiguous_size_);
         i != sparse_entries_.end() && *i == contiguous_size_;) {
      ++contiguous_size_;
      i = sparse_entries_.erase(i);
    }
  } else {
    // It's not contiguous, put it with the other sparse entries.
    CHECK(sparse_entries_.insert(sequence_number).second)
        << "sequence number " << sequence_number << " already assigned.";
  }
}


}  // namespace cert_trans
