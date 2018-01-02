#include "log/leveldb_db.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdint.h>
#include <map>
#include <string>

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
using std::string;
using std::unique_lock;
using std::unique_ptr;

DEFINE_int32(leveldb_max_open_files, 0,
             "number of open files that can be used by leveldb");
DEFINE_int32(leveldb_bloom_filter_bits_per_key, 0,
             "number of open files that can be used by leveldb");

namespace cert_trans {
namespace {


static Latency<milliseconds, string> latency_by_op_ms(
    "leveldb_latency_by_operation_ms", "operation",
    "Database latency in ms broken out by operation.");


const char kMetaNodeIdKey[] = "metadata";
const char kEntryPrefix[] = "entry-";
const char kTreeHeadPrefix[] = "sth-";
const char kMetaPrefix[] = "meta-";


#ifdef HAVE_LEVELDB_FILTER_POLICY_H
unique_ptr<const leveldb::FilterPolicy> BuildFilterPolicy() {
  unique_ptr<const leveldb::FilterPolicy> retval;

  if (FLAGS_leveldb_bloom_filter_bits_per_key > 0) {
    retval.reset(CHECK_NOTNULL(leveldb::NewBloomFilterPolicy(
        FLAGS_leveldb_bloom_filter_bits_per_key)));
  }

  return retval;
}
#endif


// WARNING: Do NOT change the type of "index" from int64_t, or you'll
// break existing databases!
string IndexToKey(int64_t index) {
  const char nibble[] = "0123456789abcdef";
  string index_str(sizeof(index) * 2, nibble[0]);
  for (int i = sizeof(index) * 2; i > 0 && index > 0; --i) {
    index_str[i - 1] = nibble[index & 0xf];
    index = index >> 4;
  }

  return kEntryPrefix + index_str;
}


int64_t KeyToIndex(leveldb::Slice key) {
  CHECK(key.starts_with(kEntryPrefix));
  key.remove_prefix(strlen(kEntryPrefix));
  const string index_str(util::BinaryString(key.ToString()));

  int64_t index(0);
  CHECK_EQ(index_str.size(), sizeof(index));
  for (size_t i = 0; i < sizeof(index); ++i) {
    index = (index << 8) | static_cast<unsigned char>(index_str[i]);
  }

  return index;
}


}  // namespace


class LevelDB::Iterator : public Database::Iterator {
 public:
  Iterator(const LevelDB* db, int64_t start_index)
      : it_(CHECK_NOTNULL(db)->db_->NewIterator(leveldb::ReadOptions())) {
    CHECK(it_);
    it_->Seek(IndexToKey(start_index));
  }

  bool GetNextEntry(LoggedEntry* entry) override {
    if (!it_->Valid() || !it_->key().starts_with(kEntryPrefix)) {
      return false;
    }

    const int64_t seq(KeyToIndex(it_->key()));
    CHECK(entry->ParseFromArray(it_->value().data(), it_->value().size()))
        << "failed to parse entry for key " << it_->key().ToString();
    CHECK(entry->has_sequence_number())
        << "no sequence number for entry with expected sequence number "
        << seq;
    CHECK_EQ(entry->sequence_number(), seq) << "unexpected sequence_number";

    it_->Next();

    return true;
  }

 private:
  const unique_ptr<leveldb::Iterator> it_;
};


const size_t LevelDB::kTimestampBytesIndexed = 6;


LevelDB::LevelDB(const string& dbfile)
    :
#ifdef HAVE_LEVELDB_FILTER_POLICY_H
      filter_policy_(BuildFilterPolicy()),
#endif
      contiguous_size_(0),
      latest_tree_timestamp_(0) {
  LOG(INFO) << "Opening " << dbfile;
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("open"));
  leveldb::Options options;
  options.create_if_missing = true;
  if (FLAGS_leveldb_max_open_files > 0) {
    options.max_open_files = FLAGS_leveldb_max_open_files;
  }
#ifdef HAVE_LEVELDB_FILTER_POLICY_H
  options.filter_policy = filter_policy_.get();
#else
  CHECK_EQ(FLAGS_leveldb_bloom_filter_bits_per_key, 0)
      << "this version of leveldb does not have bloom filter support";
#endif
  leveldb::DB* db;
  leveldb::Status status(leveldb::DB::Open(options, dbfile, &db));
  CHECK(status.ok()) << status.ToString();
  db_.reset(db);

  BuildIndex();
}


Database::WriteResult LevelDB::CreateSequencedEntry_(
    const LoggedEntry& logged) {
  CHECK(logged.has_sequence_number());
  CHECK_GE(logged.sequence_number(), 0);
  ScopedLatency latency(
      latency_by_op_ms.GetScopedLatency("create_sequenced_entry"));

  unique_lock<mutex> lock(lock_);

  string data;
  CHECK(logged.SerializeToString(&data));

  const string key(IndexToKey(logged.sequence_number()));

  string existing_data;
  leveldb::Status status(
      db_->Get(leveldb::ReadOptions(), key, &existing_data));
  if (status.IsNotFound()) {
    status = db_->Put(leveldb::WriteOptions(), key, data);
    CHECK(status.ok()) << "Failed to write sequenced entry (seq: "
                       << logged.sequence_number()
                       << "): " << status.ToString();
  } else {
    if (existing_data == data) {
      return this->OK;
    }
    return this->SEQUENCE_NUMBER_ALREADY_IN_USE;
  }

  InsertEntryMapping(logged.sequence_number(), logged.Hash());

  return this->OK;
}


Database::LookupResult LevelDB::LookupByHash(const string& hash,
                                             LoggedEntry* result) const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("lookup_by_hash"));

  unique_lock<mutex> lock(lock_);

  auto i(id_by_hash_.find(hash));
  if (i == id_by_hash_.end()) {
    return this->NOT_FOUND;
  }

  string cert_data;
  const leveldb::Status status(
      db_->Get(leveldb::ReadOptions(), IndexToKey(i->second), &cert_data));
  if (status.IsNotFound()) {
    return this->NOT_FOUND;
  }
  CHECK(status.ok()) << "Failed to get entry by hash(" << util::HexString(hash)
                     << "): " << status.ToString();

  LoggedEntry logged;
  CHECK(logged.ParseFromString(cert_data));
  CHECK_EQ(logged.Hash(), hash);

  if (result) {
    logged.Swap(result);
  }

  return this->LOOKUP_OK;
}


Database::LookupResult LevelDB::LookupByIndex(int64_t sequence_number,
                                              LoggedEntry* result) const {
  CHECK_GE(sequence_number, 0);
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("lookup_by_index"));

  string cert_data;
  leveldb::Status status(db_->Get(leveldb::ReadOptions(),
                                  IndexToKey(sequence_number), &cert_data));
  if (status.IsNotFound()) {
    return this->NOT_FOUND;
  }
  CHECK(status.ok()) << "Failed to get entry for sequence number "
                     << sequence_number;

  if (result) {
    CHECK(result->ParseFromString(cert_data));
    CHECK_EQ(result->sequence_number(), sequence_number);
  }

  return this->LOOKUP_OK;
}


unique_ptr<Database::Iterator> LevelDB::ScanEntries(
    int64_t start_index) const {
  return unique_ptr<Iterator>(new Iterator(this, start_index));
}


Database::WriteResult LevelDB::WriteTreeHead_(const ct::SignedTreeHead& sth) {
  CHECK_GE(sth.tree_size(), 0);
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("write_tree_head"));

  // 6 bytes are good enough for some 9000 years.
  string timestamp_key =
      Serializer::SerializeUint(sth.timestamp(),
                                LevelDB::kTimestampBytesIndexed);
  string data;
  CHECK(sth.SerializeToString(&data));

  unique_lock<mutex> lock(lock_);
  string existing_data;
  leveldb::Status status(db_->Get(leveldb::ReadOptions(),
                                  kTreeHeadPrefix + timestamp_key,
                                  &existing_data));
  if (status.ok()) {
    if (existing_data == data) {
      return this->OK;
    }
    return this->DUPLICATE_TREE_HEAD_TIMESTAMP;
  }

  leveldb::WriteOptions opts;
  opts.sync = true;
  status = db_->Put(opts, kTreeHeadPrefix + timestamp_key, data);
  CHECK(status.ok()) << "Failed to write tree head (" << timestamp_key
                     << "): " << status.ToString();

  if (sth.timestamp() > latest_tree_timestamp_) {
    latest_tree_timestamp_ = sth.timestamp();
    latest_timestamp_key_ = timestamp_key;
  }

  lock.unlock();
  callbacks_.Call(sth);

  return this->OK;
}


Database::LookupResult LevelDB::LatestTreeHead(
    ct::SignedTreeHead* result) const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("latest_tree_head"));
  lock_guard<mutex> lock(lock_);

  return LatestTreeHeadNoLock(result);
}


int64_t LevelDB::TreeSize() const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("tree_size"));
  lock_guard<mutex> lock(lock_);

  return contiguous_size_;
}


void LevelDB::AddNotifySTHCallback(
    const Database::NotifySTHCallback* callback) {
  unique_lock<mutex> lock(lock_);

  callbacks_.Add(callback);

  ct::SignedTreeHead sth;
  if (LatestTreeHeadNoLock(&sth) == this->LOOKUP_OK) {
    lock.unlock();
    (*callback)(sth);
  }
}


void LevelDB::RemoveNotifySTHCallback(
    const Database::NotifySTHCallback* callback) {
  lock_guard<mutex> lock(lock_);

  callbacks_.Remove(callback);
}


void LevelDB::InitializeNode(const string& node_id) {
  CHECK(!node_id.empty());
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("initialize_node"));
  unique_lock<mutex> lock(lock_);
  string existing_id;
  leveldb::Status status(db_->Get(leveldb::ReadOptions(),
                                  string(kMetaPrefix) + kMetaNodeIdKey,
                                  &existing_id));
  if (!status.IsNotFound()) {
    LOG(FATAL) << "Attempting to initialize DB beloging to node with node_id: "
               << existing_id;
  }
  status = db_->Put(leveldb::WriteOptions(),
                    string(kMetaPrefix) + kMetaNodeIdKey, node_id);
  CHECK(status.ok()) << "Failed to store NodeId: " << status.ToString();
}


Database::LookupResult LevelDB::NodeId(string* node_id) {
  CHECK_NOTNULL(node_id);
  if (!db_->Get(leveldb::ReadOptions(), string(kMetaPrefix) + kMetaNodeIdKey,
                node_id)
           .ok()) {
    return this->NOT_FOUND;
  }
  return this->LOOKUP_OK;
}


void LevelDB::BuildIndex() {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("build_index"));
  // Technically, this should only be called from the constructor, so
  // this should not be necessarily, but just to be sure...
  lock_guard<mutex> lock(lock_);

  leveldb::ReadOptions options;
  options.fill_cache = false;
  unique_ptr<leveldb::Iterator> it(db_->NewIterator(options));
  CHECK(it);
  it->Seek(kEntryPrefix);

  for (; it->Valid() && it->key().starts_with(kEntryPrefix); it->Next()) {
    const int64_t seq(KeyToIndex(it->key()));
    LoggedEntry logged;
    CHECK(logged.ParseFromString(it->value().ToString()))
        << "Failed to parse entry with sequence number " << seq;
    CHECK(logged.has_sequence_number())
        << "No sequence number for entry with sequence number " << seq;
    CHECK_EQ(logged.sequence_number(), seq)
        << "Entry has unexpected sequence_number: " << seq;

    InsertEntryMapping(logged.sequence_number(), logged.Hash());
  }

  // Now read the STH entries.
  it->Seek(kTreeHeadPrefix);
  for (; it->Valid() && it->key().starts_with(kTreeHeadPrefix); it->Next()) {
    leveldb::Slice key_slice(it->key());
    key_slice.remove_prefix(strlen(kTreeHeadPrefix));
    latest_timestamp_key_ = key_slice.ToString();
    CHECK_EQ(DeserializeResult::OK,
             Deserializer::DeserializeUint<uint64_t>(
                 latest_timestamp_key_, LevelDB::kTimestampBytesIndexed,
                 &latest_tree_timestamp_));
  }
}


Database::LookupResult LevelDB::LatestTreeHeadNoLock(
    ct::SignedTreeHead* result) const {
  if (latest_tree_timestamp_ == 0) {
    return this->NOT_FOUND;
  }

  string tree_data;
  leveldb::Status status(db_->Get(leveldb::ReadOptions(),
                                  kTreeHeadPrefix + latest_timestamp_key_,
                                  &tree_data));
  CHECK(status.ok()) << "Failed to read latest tree head: "
                     << status.ToString();

  CHECK(result->ParseFromString(tree_data));
  CHECK_EQ(result->timestamp(), latest_tree_timestamp_);

  return this->LOOKUP_OK;
}


// This must be called with "lock_" held.
void LevelDB::InsertEntryMapping(int64_t sequence_number, const string& hash) {
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
