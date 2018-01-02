#include "log/etcd_consistent_store.h"

#include <gflags/gflags.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "log/logged_entry.h"
#include "monitoring/registry.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "util/fake_etcd.h"
#include "util/libevent_wrapper.h"
#include "util/mock_masterelection.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/thread_pool.h"
#include "util/util.h"

DECLARE_int32(node_state_ttl_seconds);
DECLARE_int32(etcd_stats_collection_interval_seconds);

namespace cert_trans {


using ct::SequenceMapping;
using ct::SignedTreeHead;
using std::atomic;
using std::bind;
using std::chrono::milliseconds;
using std::lock_guard;
using std::make_pair;
using std::make_shared;
using std::mutex;
using std::ostringstream;
using std::pair;
using std::placeholders::_1;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using testing::_;
using testing::AllOf;
using testing::ContainerEq;
using testing::Contains;
using testing::Pair;
using testing::Return;
using testing::SetArgumentPointee;
using util::Status;
using util::StatusOr;
using util::SyncTask;
using util::testing::StatusIs;


const char kRoot[] = "/root";
const char kNodeId[] = "node_id";
const int kTimestamp = 9000;


class EtcdConsistentStoreTest : public ::testing::Test {
 public:
  EtcdConsistentStoreTest()
      : base_(make_shared<libevent::Base>()),
        executor_(2),
        event_pump_(base_),
        client_(base_.get()) {
  }

 protected:
  void SetUp() override {
    Registry::Instance()->ResetForTestingOnly();
    FLAGS_etcd_stats_collection_interval_seconds = 1;
    store_.reset(new EtcdConsistentStore<LoggedEntry>(base_.get(), &executor_,
                                                      &client_, &election_,
                                                      kRoot, kNodeId));
    InsertEntry("/root/sequence_mapping", SequenceMapping());
  }

  LoggedEntry DefaultCert() {
    return MakeCert(kTimestamp, "leaf");
  }

  LoggedEntry MakeCert(int timestamp, const string& body) {
    LoggedEntry cert;
    cert.mutable_sct()->set_timestamp(timestamp);
    cert.mutable_entry()->set_type(ct::X509_ENTRY);
    cert.mutable_entry()->mutable_x509_entry()->set_leaf_certificate(body);
    return cert;
  }

  LoggedEntry MakeSequencedCert(int timestamp, const string& body, int seq) {
    LoggedEntry cert(MakeCert(timestamp, body));
    cert.set_sequence_number(seq);
    return cert;
  }

  string CertPath(const LoggedEntry& cert) const {
    return string(kRoot) + "/entries/ " + util::HexString(cert.Hash());
  }

  EntryHandle<LoggedEntry> HandleForCert(const LoggedEntry& cert) {
    return EntryHandle<LoggedEntry>(CertPath(cert), cert);
  }

  EntryHandle<LoggedEntry> HandleForCert(const LoggedEntry& cert, int handle) {
    return EntryHandle<LoggedEntry>(CertPath(cert), cert, handle);
  }

  void PopulateForCleanupTests(int num_seq, int num_pending,
                               int starting_seq) {
    int timestamp(345345);
    int seq(starting_seq);
    EntryHandle<SequenceMapping> mapping;
    Status status(store_->GetSequenceMapping(&mapping));
    CHECK_EQ(Status::OK, status);
    for (int i = 0; i < num_seq; ++i) {
      std::ostringstream ss;
      ss << "sequenced body " << i;
      LoggedEntry lc(MakeCert(timestamp++, ss.str()));
      CHECK(store_->AddPendingEntry(&lc).ok());
      SequenceMapping::Mapping* m(mapping.MutableEntry()->add_mapping());
      m->set_entry_hash(lc.Hash());
      m->set_sequence_number(seq++);
    }
    CHECK_EQ(Status::OK, store_->UpdateSequenceMapping(&mapping));
    for (int i = 0; i < num_pending; ++i) {
      std::ostringstream ss;
      ss << "pending body " << i;
      LoggedEntry lc(MakeCert(timestamp++, ss.str()));
      CHECK(store_->AddPendingEntry(&lc).ok());
    }
  }

  util::StatusOr<int64_t> CleanupOldEntries() {
    return store_->CleanupOldEntries();
  }

  void AddSequenceMapping(int64_t seq, const string& hash) {
    EntryHandle<SequenceMapping> mapping;
    Status status(store_->GetSequenceMapping(&mapping));
    CHECK_EQ(Status::OK, status);
    SequenceMapping::Mapping* m(mapping.MutableEntry()->add_mapping());
    m->set_sequence_number(seq);
    m->set_entry_hash(hash);
    CHECK_EQ(Status::OK, store_->UpdateSequenceMapping(&mapping));
  }


  template <class T>
  void ForceSetEntry(const string& key, const T& thing) {
    // Set up scenario:
    SyncTask task(base_.get());
    EtcdClient::Response resp;
    client_.ForceSet(key, Serialize(thing), &resp, task.task());
    task.Wait();
    ASSERT_EQ(Status::OK, task.status());
  }


  template <class T>
  void InsertEntry(const string& key, const T& thing) {
    // Set up scenario:
    SyncTask task(base_.get());
    EtcdClient::Response resp;
    client_.Create(key, Serialize(thing), &resp, task.task());
    task.Wait();
    ASSERT_EQ(Status::OK, task.status());
  }

  template <class T>
  void PeekEntry(const string& key, T* thing) {
    EtcdClient::GetResponse resp;
    SyncTask task(base_.get());
    client_.Get(key, &resp, task.task());
    task.Wait();
    ASSERT_EQ(Status::OK, task.status());
    Deserialize(resp.node.value_, thing);
  }

  template <class T>
  string Serialize(const T& t) {
    string flat;
    t.SerializeToString(&flat);
    return util::ToBase64(flat);
  }

  template <class T>
  void Deserialize(const string& flat, T* t) {
    ASSERT_TRUE(t->ParseFromString(util::FromBase64(flat.c_str())));
  }

  template <class T>
  EtcdClient::Node NodeFor(const int index, const std::string& key,
                           const T& t) {
    return EtcdClient::Node(index, index, key, Serialize(t));
  }

  ct::SignedTreeHead ServingSTH() {
    return store_->serving_sth_->Entry();
  }

  int64_t GetNumEtcdEntries() const {
    return store_->num_etcd_entries_;
  }


  shared_ptr<libevent::Base> base_;
  ThreadPool executor_;
  libevent::EventPumpThread event_pump_;
  FakeEtcdClient client_;
  MockMasterElection election_;
  unique_ptr<EtcdConsistentStore<LoggedEntry>> store_;
};


typedef class EtcdConsistentStoreTest EtcdConsistentStoreDeathTest;


TEST_F(
    EtcdConsistentStoreDeathTest,
    TestNextAvailableSequenceNumberWhenNoSequencedEntriesOrServingSTHExist) {
  util::StatusOr<int64_t> sequence_number(
      store_->NextAvailableSequenceNumber());
  ASSERT_EQ(Status::OK, sequence_number.status());
  EXPECT_EQ(0, sequence_number.ValueOrDie());
}


TEST_F(EtcdConsistentStoreTest,
       TestNextAvailableSequenceNumberWhenSequencedEntriesExist) {
  AddSequenceMapping(0, "zero");
  AddSequenceMapping(1, "one");
  util::StatusOr<int64_t> sequence_number(
      store_->NextAvailableSequenceNumber());
  ASSERT_EQ(Status::OK, sequence_number.status());
  EXPECT_EQ(2, sequence_number.ValueOrDie());
}


TEST_F(EtcdConsistentStoreTest,
       TestNextAvailableSequenceNumberWhenNoSequencedEntriesExistButHaveSTH) {
  ct::SignedTreeHead serving_sth;
  serving_sth.set_timestamp(123);
  serving_sth.set_tree_size(600);
  EXPECT_TRUE(store_->SetServingSTH(serving_sth).ok());

  util::StatusOr<int64_t> sequence_number(
      store_->NextAvailableSequenceNumber());
  ASSERT_EQ(Status::OK, sequence_number.status());
  EXPECT_EQ(serving_sth.tree_size(), sequence_number.ValueOrDie());
}


TEST_F(EtcdConsistentStoreTest, TestSetServingSTH) {
  ct::SignedTreeHead sth;
  util::Status status(store_->SetServingSTH(sth));
  EXPECT_TRUE(status.ok()) << status;
}


TEST_F(EtcdConsistentStoreTest, TestSetServingSTHOverwrites) {
  ct::SignedTreeHead sth;
  sth.set_timestamp(234);
  util::Status status(store_->SetServingSTH(sth));
  EXPECT_TRUE(status.ok()) << status;

  ct::SignedTreeHead sth2;
  sth2.set_timestamp(sth.timestamp() + 1);
  status = store_->SetServingSTH(sth2);
  EXPECT_TRUE(status.ok()) << status;
}


TEST_F(EtcdConsistentStoreTest, TestSetServingSTHWontOverwriteWithOlder) {
  ct::SignedTreeHead sth;
  sth.set_timestamp(234);
  EXPECT_OK(store_->SetServingSTH(sth));

  ct::SignedTreeHead sth2;
  sth2.set_timestamp(sth.timestamp() - 1);
  EXPECT_THAT(store_->SetServingSTH(sth2),
              StatusIs(util::error::OUT_OF_RANGE));
}

TEST_F(EtcdConsistentStoreDeathTest, TestSetServingSTHChecksInconsistentSize) {
  ct::SignedTreeHead sth;
  sth.set_timestamp(234);
  sth.set_tree_size(10);
  util::Status status(store_->SetServingSTH(sth));
  EXPECT_TRUE(status.ok()) << status;

  ct::SignedTreeHead sth2;
  // newer STH...
  sth2.set_timestamp(sth.timestamp() + 1);
  // but. curiously, a smaller tree...
  sth2.set_tree_size(sth.tree_size() - 1);
  EXPECT_DEATH(store_->SetServingSTH(sth2), "tree_size");
}


TEST_F(EtcdConsistentStoreTest, TestAddPendingEntryWorks) {
  LoggedEntry cert(DefaultCert());
  util::Status status(store_->AddPendingEntry(&cert));
  ASSERT_EQ(Status::OK, status);
  EtcdClient::GetResponse resp;
  SyncTask task(base_.get());
  client_.Get(string(kRoot) + "/entries/" + util::HexString(cert.Hash()),
              &resp, task.task());
  task.Wait();
  EXPECT_EQ(Status::OK, task.status());
  EXPECT_EQ(Serialize(cert), resp.node.value_);
}


TEST_F(EtcdConsistentStoreTest,
       TestAddPendingEntryForExistingEntryReturnsSct) {
  LoggedEntry cert(DefaultCert());
  LoggedEntry other_cert(DefaultCert());
  other_cert.mutable_sct()->set_timestamp(55555);

  const string kKey(util::HexString(cert.Hash()));
  const string kPath(string(kRoot) + "/entries/" + kKey);
  // Set up scenario:
  InsertEntry(kPath, other_cert);

  EXPECT_THAT(store_->AddPendingEntry(&cert),
              StatusIs(util::error::ALREADY_EXISTS));
  EXPECT_EQ(other_cert.timestamp(), cert.timestamp());
}


TEST_F(EtcdConsistentStoreDeathTest,
       TestAddPendingEntryForExistingNonIdenticalEntry) {
  LoggedEntry cert(DefaultCert());
  LoggedEntry other_cert(MakeCert(2342, "something else"));

  const string kKey(util::HexString(cert.Hash()));
  const string kPath(string(kRoot) + "/entries/" + kKey);
  // Set up scenario:
  InsertEntry(kPath, other_cert);

  EXPECT_DEATH(store_->AddPendingEntry(&cert),
               "Check failed: LeafEntriesMatch");
}


TEST_F(EtcdConsistentStoreDeathTest,
       TestAddPendingEntryDoesNotAcceptSequencedEntry) {
  LoggedEntry cert(DefaultCert());
  cert.set_sequence_number(76);
  EXPECT_DEATH(store_->AddPendingEntry(&cert),
               "!entry\\->has_sequence_number");
}


TEST_F(EtcdConsistentStoreTest, TestGetPendingEntryForHash) {
  const LoggedEntry one(MakeCert(123, "one"));
  const string kPath(string(kRoot) + "/entries/" +
                     util::HexString(one.Hash()));
  InsertEntry(kPath, one);

  EntryHandle<LoggedEntry> handle;
  util::Status status(store_->GetPendingEntryForHash(one.Hash(), &handle));
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_EQ(one, handle.Entry());
}


TEST_F(EtcdConsistentStoreTest, TestGetPendingEntryForNonExistantHash) {
  const string kPath(string(kRoot) + "/entries/" + util::HexString("Nah"));
  EntryHandle<LoggedEntry> handle;
  EXPECT_THAT(store_->GetPendingEntryForHash("Nah", &handle),
              StatusIs(util::error::NOT_FOUND));
}


TEST_F(EtcdConsistentStoreTest, TestGetPendingEntries) {
  const string kPath(string(kRoot) + "/entries/");
  const LoggedEntry one(MakeCert(123, "one"));
  const LoggedEntry two(MakeCert(456, "two"));
  InsertEntry(kPath + "one", one);
  InsertEntry(kPath + "two", two);

  vector<EntryHandle<LoggedEntry>> entries;
  util::Status status(store_->GetPendingEntries(&entries));
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_EQ(static_cast<size_t>(2), entries.size());
  vector<LoggedEntry> certs;
  for (const auto& e : entries) {
    certs.push_back(e.Entry());
  }
  EXPECT_THAT(certs, AllOf(Contains(one), Contains(two)));
}


TEST_F(EtcdConsistentStoreDeathTest,
       TestGetPendingEntriesBarfsWithSequencedEntry) {
  const string kPath(string(kRoot) + "/entries/");
  LoggedEntry one(MakeSequencedCert(123, "one", 666));
  InsertEntry(kPath + "one", one);
  vector<EntryHandle<LoggedEntry>> entries;
  EXPECT_DEATH(store_->GetPendingEntries(&entries), "has_sequence_number");
}


TEST_F(EtcdConsistentStoreTest, TestGetSequenceMapping) {
  AddSequenceMapping(0, "zero");
  AddSequenceMapping(1, "one");
  EntryHandle<SequenceMapping> mapping;
  util::Status status(store_->GetSequenceMapping(&mapping));
  CHECK_EQ(util::Status::OK, status);

  EXPECT_EQ(2, mapping.Entry().mapping_size());
  EXPECT_EQ(0, mapping.Entry().mapping(0).sequence_number());
  EXPECT_EQ("zero", mapping.Entry().mapping(0).entry_hash());
  EXPECT_EQ(1, mapping.Entry().mapping(1).sequence_number());
  EXPECT_EQ("one", mapping.Entry().mapping(1).entry_hash());
}


TEST_F(EtcdConsistentStoreTest,
       TestGetSequenceMappingAllowsGapsBelowTreeSize) {
  SignedTreeHead sth;
  sth.set_tree_size(3);
  store_->SetServingSTH(sth);

  SequenceMapping mapping;
  mapping.add_mapping()->set_sequence_number(0);
  mapping.add_mapping()->set_sequence_number(2);
  ForceSetEntry("/root/sequence_mapping", mapping);
  EntryHandle<SequenceMapping> entry;
  EXPECT_OK(store_->GetSequenceMapping(&entry));
}


TEST_F(EtcdConsistentStoreDeathTest,
       TestGetSequenceMappingBarfsOnGapsAboveTreeSize) {
  SignedTreeHead sth;
  sth.set_tree_size(0);
  store_->SetServingSTH(sth);

  SequenceMapping mapping;
  mapping.add_mapping()->set_sequence_number(0);
  mapping.add_mapping()->set_sequence_number(2);
  ForceSetEntry("/root/sequence_mapping", mapping);
  EntryHandle<SequenceMapping> entry;
  EXPECT_DEATH(store_->GetSequenceMapping(&entry), "mapped_seq \\+ 1");
}


TEST_F(EtcdConsistentStoreTest, TestUpdateSequenceMapping) {
  EntryHandle<SequenceMapping> mapping;
  Status status(store_->GetSequenceMapping(&mapping));
  EXPECT_EQ(Status::OK, status);

  const SequenceMapping original(mapping.Entry());

  SequenceMapping::Mapping* m(mapping.MutableEntry()->add_mapping());
  m->set_sequence_number(0);
  m->set_entry_hash("zero");
  EXPECT_EQ(Status::OK, store_->UpdateSequenceMapping(&mapping));

  status = store_->GetSequenceMapping(&mapping);
  EXPECT_EQ(Status::OK, status);

  EXPECT_EQ(mapping.Entry().mapping_size(), original.mapping_size() + 1);
  EXPECT_EQ(0, mapping.Entry()
                   .mapping(mapping.Entry().mapping_size() - 1)
                   .sequence_number());
  EXPECT_EQ("zero", mapping.Entry()
                        .mapping(mapping.Entry().mapping_size() - 1)
                        .entry_hash());
}


TEST_F(EtcdConsistentStoreDeathTest,
       TestUpdateSequenceMappingBarfsWithOutOfOrderSequenceNumber) {
  EntryHandle<SequenceMapping> mapping;
  Status status(store_->GetSequenceMapping(&mapping));
  EXPECT_EQ(Status::OK, status);

  SequenceMapping::Mapping* m1(mapping.MutableEntry()->add_mapping());
  m1->set_sequence_number(2);
  m1->set_entry_hash("two");
  SequenceMapping::Mapping* m2(mapping.MutableEntry()->add_mapping());
  m2->set_sequence_number(0);
  m2->set_entry_hash("zero");
  EXPECT_DEATH(store_->UpdateSequenceMapping(&mapping),
               "sequence_number\\(\\) < mapping");
}

TEST_F(EtcdConsistentStoreDeathTest,
       TestUpdateSequenceMappingBarfsMappingNonContiguousToServingTree) {
  SignedTreeHead sth;
  sth.set_timestamp(123);
  sth.set_tree_size(1000);
  CHECK_EQ(Status::OK, store_->SetServingSTH(sth));

  EntryHandle<SequenceMapping> mapping;
  Status status(store_->GetSequenceMapping(&mapping));
  EXPECT_EQ(Status::OK, status);

  SequenceMapping::Mapping* m1(mapping.MutableEntry()->add_mapping());
  m1->set_sequence_number(sth.tree_size() + 1);
  m1->set_entry_hash("zero");
  EXPECT_DEATH(store_->UpdateSequenceMapping(&mapping),
               "lowest_sequence_number <= tree_size");
}


TEST_F(EtcdConsistentStoreTest, TestSetClusterNodeState) {
  const string kPath(string(kRoot) + "/nodes/" + kNodeId);

  ct::ClusterNodeState state;
  state.set_node_id(kNodeId);

  util::Status status(store_->SetClusterNodeState(state));
  EXPECT_TRUE(status.ok()) << status;

  ct::ClusterNodeState set_state;
  PeekEntry(kPath, &set_state);
  EXPECT_EQ(state.node_id(), set_state.node_id());
}


TEST_F(EtcdConsistentStoreTest, TestSetClusterNodeStateHasTTL) {
  FLAGS_node_state_ttl_seconds = 1;
  const string kPath(string(kRoot) + "/nodes/" + kNodeId);

  ct::ClusterNodeState state;
  state.set_node_id(kNodeId);

  util::Status status(store_->SetClusterNodeState(state));
  EXPECT_TRUE(status.ok()) << status;

  ct::ClusterNodeState set_state;
  PeekEntry(kPath, &set_state);
  EXPECT_EQ(state.node_id(), set_state.node_id());

  sleep(2);

  EtcdClient::GetResponse resp;
  SyncTask task(base_.get());
  client_.Get(kPath, &resp, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::NOT_FOUND));
}


TEST_F(EtcdConsistentStoreTest, WatchServingSTH) {
  Notification notify;

  const string kPath(string(kRoot) + "/serving_sth");

  ct::SignedTreeHead sth;
  sth.set_timestamp(234234);

  SyncTask task(&executor_);
  mutex mutex;
  int call_count(0);
  store_->WatchServingSTH(
      [&sth, &notify, &call_count,
       &mutex](const Update<ct::SignedTreeHead>& update) {
        lock_guard<std::mutex> lock(mutex);
        ASSERT_LE(call_count, 2)
            << "Extra update: key:" << update.handle_.Key() << "@"
            << update.handle_.Handle() << " exists:" << update.exists_
            << " entry: " << update.handle_.Entry().DebugString();
        switch (call_count) {
          case 0:
            // initial empty state
            EXPECT_FALSE(update.exists_);
            break;
          case 1:
            // notification of update
            EXPECT_TRUE(update.exists_);
            EXPECT_EQ(sth.DebugString(), update.handle_.Entry().DebugString());
            notify.Notify();
            break;
        }
        ++call_count;
      },
      task.task());

  util::Status status(store_->SetServingSTH(sth));
  EXPECT_TRUE(status.ok()) << status;
  notify.WaitForNotification();
  EXPECT_EQ(ServingSTH().DebugString(), sth.DebugString());
  task.Cancel();
  task.Wait();
}


TEST_F(EtcdConsistentStoreTest, WatchClusterNodeStates) {
  const string kPath(string(kRoot) + "/nodes/" + kNodeId);

  ct::ClusterNodeState state;
  state.set_node_id(kNodeId);

  SyncTask task(&executor_);
  store_->WatchClusterNodeStates(
      [&state](const vector<Update<ct::ClusterNodeState>>& updates) {
        if (updates.empty()) {
          VLOG(1) << "Ignoring initial empty update.";
          return;
        }
        EXPECT_TRUE(updates[0].exists_);
        EXPECT_EQ(updates[0].handle_.Entry().DebugString(),
                  state.DebugString());
      },
      task.task());
  util::Status status(store_->SetClusterNodeState(state));
  EXPECT_TRUE(status.ok()) << status;
  task.Cancel();
  task.Wait();
}


TEST_F(EtcdConsistentStoreTest, WatchClusterConfig) {
  const string kPath(string(kRoot) + "/cluster_config");

  ct::ClusterConfig config;
  config.set_minimum_serving_nodes(1);
  config.set_minimum_serving_fraction(0.6);
  Notification notification;

  SyncTask task(&executor_);
  store_->WatchClusterConfig(
      [&config, &notification](const Update<ct::ClusterConfig>& update) {
        if (!update.exists_) {
          VLOG(1) << "Ignoring initial empty update.";
          return;
        }
        EXPECT_TRUE(update.exists_);
        EXPECT_EQ(update.handle_.Entry().DebugString(), config.DebugString());
        notification.Notify();
      },
      task.task());
  util::Status status(store_->SetClusterConfig(config));
  EXPECT_TRUE(status.ok()) << status;
  // Make sure we got called from the watcher:
  EXPECT_TRUE(notification.WaitForNotificationWithTimeout(milliseconds(5000)));
  task.Cancel();
  task.Wait();
}


TEST_F(EtcdConsistentStoreTest, TestDoesNotCleanUpIfNotMaster) {
  EXPECT_CALL(election_, IsMaster()).WillRepeatedly(Return(false));
  EXPECT_THAT(CleanupOldEntries().status(),
              StatusIs(util::error::PERMISSION_DENIED));
}


TEST_F(EtcdConsistentStoreTest, TestEmptyClean) {
  EXPECT_CALL(election_, IsMaster()).WillRepeatedly(Return(true));
  const StatusOr<int64_t> num_cleaned(CleanupOldEntries());
  ASSERT_OK(num_cleaned.status());
  EXPECT_EQ(0, num_cleaned.ValueOrDie());
}


TEST_F(EtcdConsistentStoreTest, TestCleansUpToNewSTH) {
  PopulateForCleanupTests(5, 4, 100);

  // Be sure about our starting state of sequenced entries so we can compare
  // later on
  EntryHandle<SequenceMapping> orig_seq_mapping;
  ASSERT_EQ(Status::OK, store_->GetSequenceMapping(&orig_seq_mapping));
  EXPECT_EQ(5, orig_seq_mapping.Entry().mapping_size());

  // Do the same for the pending entries
  vector<EntryHandle<LoggedEntry>> pending_entries_pre;
  CHECK(store_->GetPendingEntries(&pending_entries_pre).ok());
  // Prune out any "pending" entries which have counterparts in the
  // "sequenced" set:
  unordered_map<int64_t, string> seq_to_hash;
  {
    unordered_set<string> seq_hashes;
    for (auto& m : orig_seq_mapping.Entry().mapping()) {
      seq_hashes.insert(m.entry_hash());
      seq_to_hash.insert(make_pair(m.sequence_number(), m.entry_hash()));
    }
    auto it(pending_entries_pre.begin());
    while (it != pending_entries_pre.end()) {
      if (seq_hashes.find(it->Entry().Hash()) != seq_hashes.end()) {
        it = pending_entries_pre.erase(it);
      } else {
        ++it;
      }
    }
  }
  EXPECT_EQ(static_cast<size_t>(4), pending_entries_pre.size());

  EXPECT_CALL(election_, IsMaster()).WillRepeatedly(Return(true));

  // Set ServingSTH to something which will cause entries 100, 101, and 102 to
  // be cleaned up:
  SignedTreeHead sth;
  sth.set_timestamp(345345);
  sth.set_tree_size(103);
  CHECK(store_->SetServingSTH(sth).ok());
  {
    const StatusOr<int64_t> num_cleaned(CleanupOldEntries());
    ASSERT_OK(num_cleaned.status());
    EXPECT_EQ(3, num_cleaned.ValueOrDie());
  }

  EntryHandle<LoggedEntry> unused;
  EXPECT_THAT(store_->GetPendingEntryForHash(seq_to_hash[100], &unused),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_THAT(store_->GetPendingEntryForHash(seq_to_hash[101], &unused),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_THAT(store_->GetPendingEntryForHash(seq_to_hash[102], &unused),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_OK(store_->GetPendingEntryForHash(seq_to_hash[103], &unused));

  // Check that we didn't modify the sequence mapping:
  EntryHandle<SequenceMapping> seq_mapping;
  CHECK(store_->GetSequenceMapping(&seq_mapping).ok());
  EXPECT_EQ(orig_seq_mapping.Entry().DebugString(),
            seq_mapping.Entry().DebugString());

  // Now update ServingSTH so that all sequenced entries should be cleaned up:
  sth.set_timestamp(sth.timestamp() + 1);
  sth.set_tree_size(105);
  CHECK(store_->SetServingSTH(sth).ok());
  {
    const StatusOr<int64_t> num_cleaned(CleanupOldEntries());
    ASSERT_OK(num_cleaned.status());
    EXPECT_EQ(5, num_cleaned.ValueOrDie());
  }


  // Ensure they were:
  EXPECT_THAT(store_->GetPendingEntryForHash(seq_to_hash[103], &unused),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_THAT(store_->GetPendingEntryForHash(seq_to_hash[104], &unused),
              StatusIs(util::error::NOT_FOUND));

  // Check that we didn't modify the sequence mapping:
  CHECK(store_->GetSequenceMapping(&seq_mapping).ok());
  EXPECT_EQ(orig_seq_mapping.Entry().DebugString(),
            seq_mapping.Entry().DebugString());

  // Check we've not touched the pending entries:
  vector<EntryHandle<LoggedEntry>> pending_entries_post;
  CHECK(store_->GetPendingEntries(&pending_entries_post).ok());
  EXPECT_EQ(pending_entries_pre.size(), pending_entries_post.size());
  for (int i = 0; i < static_cast<int>(pending_entries_pre.size()); ++i) {
    EXPECT_EQ(pending_entries_pre[i].Handle(),
              pending_entries_post[i].Handle());
    EXPECT_EQ(pending_entries_pre[i].Entry(), pending_entries_post[i].Entry());
  }
}


TEST_F(EtcdConsistentStoreTest, TestStoreStatsFetcher) {
  EXPECT_EQ(0, GetNumEtcdEntries());
  PopulateForCleanupTests(100, 100, 100);
  sleep(2 * FLAGS_etcd_stats_collection_interval_seconds);
  EXPECT_LE(200, GetNumEtcdEntries());
}


TEST_F(EtcdConsistentStoreTest, TestRejectsAddsWhenOverCapacity) {
  ct::ClusterConfig config;
  config.set_etcd_reject_add_pending_threshold(2);
  util::Status status(store_->SetClusterConfig(config));
  ASSERT_OK(status);

  EXPECT_EQ(0, GetNumEtcdEntries());

  PopulateForCleanupTests(3, 0, 1);
  sleep(2 * FLAGS_etcd_stats_collection_interval_seconds);

  EXPECT_LT(2, GetNumEtcdEntries());

  LoggedEntry cert(MakeCert(1000, "cert1000"));
  EXPECT_THAT(store_->AddPendingEntry(&cert),
              StatusIs(util::error::RESOURCE_EXHAUSTED));
}


}  // namespace cert_trans

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
