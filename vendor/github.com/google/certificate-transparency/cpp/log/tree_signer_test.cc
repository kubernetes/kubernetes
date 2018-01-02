/* -*- indent-tabs-mode: nil -*- */
#include <gtest/gtest.h>
#include <stdint.h>
#include <memory>
#include <string>

#include "log/etcd_consistent_store-inl.h"
#include "log/file_db.h"
#include "log/log_signer.h"
#include "log/log_verifier.h"
#include "log/sqlite_db.h"
#include "log/test_db.h"
#include "log/test_signer.h"
#include "log/tree_signer-inl.h"
#include "log/tree_signer.h"
#include "merkletree/merkle_verifier.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "util/fake_etcd.h"
#include "util/mock_masterelection.h"
#include "util/status_test_util.h"
#include "util/sync_task.h"
#include "util/testing.h"
#include "util/thread_pool.h"
#include "util/util.h"

namespace cert_trans {

using cert_trans::EntryHandle;
using cert_trans::LoggedEntry;
using cert_trans::MockMasterElection;
using ct::ClusterNodeState;
using ct::SequenceMapping;
using ct::SignedTreeHead;
using std::make_shared;
using std::move;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
using testing::NiceMock;
using util::Status;

typedef TreeSigner<LoggedEntry> TS;

// TODO(alcutter): figure out if/how we can keep abstract rather than
// hardcoding LoggedEntry in here.
template <class T>
class TreeSignerTest : public ::testing::Test {
 protected:
  TreeSignerTest()
      : test_db_(),
        base_(make_shared<libevent::Base>()),
        event_pump_(base_),
        etcd_client_(base_.get()),
        pool_(2),
        test_signer_(),
        verifier_(),
        tree_signer_() {
  }

  void SetUp() {
    test_db_.reset(new TestDB<T>);
    verifier_.reset(new LogVerifier(TestSigner::DefaultLogSigVerifier(),
                                    new MerkleVerifier(new Sha256Hasher())));
    store_.reset(new EtcdConsistentStore<LoggedEntry>(
        base_.get(), &pool_, &etcd_client_, &election_, "/root", "id"));
    log_signer_.reset(TestSigner::DefaultLogSigner());
    tree_signer_.reset(new TS(std::chrono::duration<double>(0), db(),
                              unique_ptr<CompactMerkleTree>(
                                  new CompactMerkleTree(new Sha256Hasher)),
                              store_.get(), log_signer_.get()));
    // Set a default empty STH so that we can call UpdateTree() on the signer.
    store_->SetServingSTH(SignedTreeHead());
    // Force an empty sequence mapping file:
    {
      util::SyncTask task(&pool_);
      EtcdClient::Response r;
      etcd_client_.ForceSet("/root/sequence_mapping", "", &r, task.task());
      task.Wait();
    }
  }

  void AddPendingEntry(LoggedEntry* logged_cert) const {
    logged_cert->clear_sequence_number();
    CHECK(this->store_->AddPendingEntry(logged_cert).ok());
  }

  void DeletePendingEntry(const LoggedEntry& logged_cert) const {
    EntryHandle<LoggedEntry> e;
    CHECK_EQ(Status::OK,
             this->store_->GetPendingEntryForHash(logged_cert.Hash(), &e));
    CHECK_EQ(Status::OK, this->store_->DeleteEntry(&e));
  }

  void AddSequencedEntry(LoggedEntry* logged_cert, int64_t seq) const {
    logged_cert->clear_sequence_number();
    CHECK(this->store_->AddPendingEntry(logged_cert).ok());

    // This below would normally be done by TreeSigner::SequenceNewEntries()
    EntryHandle<LoggedEntry> entry;
    EntryHandle<SequenceMapping> mapping;
    CHECK(this->store_->GetSequenceMapping(&mapping).ok());
    SequenceMapping::Mapping* m(mapping.MutableEntry()->add_mapping());
    m->set_sequence_number(seq);
    m->set_entry_hash(logged_cert->Hash());
    CHECK(this->store_->UpdateSequenceMapping(&mapping).ok());
    logged_cert->set_sequence_number(seq);
    CHECK_EQ(Database::OK, this->db()->CreateSequencedEntry(*logged_cert));
  }


  TS* GetSimilar() {
    return new TS(std::chrono::duration<double>(0), db(),
                  unique_ptr<CompactMerkleTree>(new CompactMerkleTree(
                      *tree_signer_->cert_tree_, new Sha256Hasher)),
                  store_.get(), log_signer_.get());
  }

  T* db() const {
    return test_db_->db();
  }
  unique_ptr<TestDB<T>> test_db_;
  shared_ptr<libevent::Base> base_;
  libevent::EventPumpThread event_pump_;
  FakeEtcdClient etcd_client_;
  ThreadPool pool_;
  NiceMock<MockMasterElection> election_;
  std::unique_ptr<EtcdConsistentStore<LoggedEntry>> store_;
  TestSigner test_signer_;
  unique_ptr<LogVerifier> verifier_;
  unique_ptr<LogSigner> log_signer_;
  unique_ptr<TS> tree_signer_;
};

typedef testing::Types<FileDB, SQLiteDB> Databases;


EntryHandle<LoggedEntry> H(const LoggedEntry& l) {
  EntryHandle<LoggedEntry> handle;
  handle.MutableEntry()->CopyFrom(l);
  return handle;
}


TYPED_TEST_CASE(TreeSignerTest, Databases);

TYPED_TEST(TreeSignerTest, PendingEntriesOrder) {
  PendingEntriesOrder<LoggedEntry> ordering;
  LoggedEntry lowest;
  this->test_signer_.CreateUnique(&lowest);

  // Can't be lower than itself!
  EXPECT_FALSE(ordering(H(lowest), H(lowest)));

  // check timestamp:
  LoggedEntry higher_timestamp(lowest);
  higher_timestamp.mutable_sct()->set_timestamp(lowest.timestamp() + 1);
  EXPECT_TRUE(ordering(H(lowest), H(higher_timestamp)));
  EXPECT_FALSE(ordering(H(higher_timestamp), H(lowest)));

  // check hash fallback:
  LoggedEntry higher_hash(lowest);
  while (higher_hash.Hash() <= lowest.Hash()) {
    this->test_signer_.CreateUnique(&higher_hash);
    higher_hash.mutable_sct()->set_timestamp(lowest.timestamp());
  }
  EXPECT_TRUE(ordering(H(lowest), H(higher_hash)));
  EXPECT_FALSE(ordering(H(higher_hash), H(lowest)));
}


// TODO(ekasper): KAT tests.
TYPED_TEST(TreeSignerTest, Sign) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->AddPendingEntry(&logged_cert);
  // this->AddSequencedEntry(&logged_cert, 0);
  EXPECT_OK(this->tree_signer_->SequenceNewEntries());
  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());

  const SignedTreeHead sth(this->tree_signer_->LatestSTH());
  EXPECT_EQ(1U, sth.tree_size());
  EXPECT_EQ(sth.timestamp(), this->tree_signer_->LastUpdateTime());
}


TYPED_TEST(TreeSignerTest, Timestamp) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->AddSequencedEntry(&logged_cert, 0);

  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());
  uint64_t last_update = this->tree_signer_->LastUpdateTime();
  EXPECT_GE(last_update, logged_cert.sct().timestamp());

  // Now create a second entry with a timestamp some time in the future
  // and verify that the signer's timestamp is greater than that.
  uint64_t future = last_update + 10000;
  LoggedEntry logged_cert2;
  this->test_signer_.CreateUnique(&logged_cert2);
  logged_cert2.mutable_sct()->set_timestamp(future);
  this->AddSequencedEntry(&logged_cert2, 1);

  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());
  EXPECT_GE(this->tree_signer_->LastUpdateTime(), future);
}


TYPED_TEST(TreeSignerTest, Verify) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->AddSequencedEntry(&logged_cert, 0);

  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());

  const SignedTreeHead sth(this->tree_signer_->LatestSTH());
  EXPECT_EQ(LogVerifier::VERIFY_OK,
            this->verifier_->VerifySignedTreeHead(sth));
}


TYPED_TEST(TreeSignerTest, ResumeClean) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->AddSequencedEntry(&logged_cert, 0);

  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());
  const SignedTreeHead sth(this->tree_signer_->LatestSTH());
  {
    // Simulate the caller of UpdateTree() pushing this new tree out to the
    // cluster.
    ClusterNodeState node_state;
    *node_state.mutable_newest_sth() = sth;
    CHECK_EQ(util::Status::OK, this->store_->SetClusterNodeState(node_state));
  }

  unique_ptr<TS> signer2(this->GetSimilar());

  // Update
  EXPECT_EQ(TS::OK, signer2->UpdateTree());

  const SignedTreeHead sth2(signer2->LatestSTH());
  EXPECT_LT(sth.timestamp(), sth2.timestamp());
  EXPECT_EQ(sth.sha256_root_hash(), sth2.sha256_root_hash());
  EXPECT_EQ(sth.tree_size(), sth2.tree_size());
}


// Test resuming when the tree head signature is lagging behind the
// sequence number commits.
TYPED_TEST(TreeSignerTest, ResumePartialSign) {
  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());
  const SignedTreeHead sth(this->tree_signer_->LatestSTH());
  {
    // Simulate the caller of UpdateTree() pushing this new tree out to the
    // cluster.
    ClusterNodeState node_state;
    *node_state.mutable_newest_sth() = sth;
    CHECK_EQ(util::Status::OK, this->store_->SetClusterNodeState(node_state));
  }

  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->AddSequencedEntry(&logged_cert, 0);

  unique_ptr<TS> signer2(this->GetSimilar());
  EXPECT_EQ(TS::OK, signer2->UpdateTree());
  const SignedTreeHead sth2(signer2->LatestSTH());
  // The signer should have picked up the sequence number commit.
  EXPECT_EQ(1U, sth2.tree_size());
  EXPECT_LT(sth.timestamp(), sth2.timestamp());
  EXPECT_NE(sth.sha256_root_hash(), sth2.sha256_root_hash());
}


TYPED_TEST(TreeSignerTest, SignEmpty) {
  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());

  const SignedTreeHead sth(this->tree_signer_->LatestSTH());
  EXPECT_GT(sth.timestamp(), 0U);
  EXPECT_EQ(sth.tree_size(), 0U);
}


TYPED_TEST(TreeSignerTest, SequenceNewEntriesCleansUpOldSequenceMappings) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->AddPendingEntry(&logged_cert);
  EXPECT_OK(this->tree_signer_->SequenceNewEntries());
  EXPECT_EQ(TS::OK, this->tree_signer_->UpdateTree());
  EXPECT_EQ(Status::OK,
            this->store_->SetServingSTH(this->tree_signer_->LatestSTH()));
  sleep(1);

  {
    EntryHandle<SequenceMapping> mapping;
    CHECK_EQ(Status::OK, this->store_->GetSequenceMapping(&mapping));
    EXPECT_EQ(1, mapping.Entry().mapping_size());
    EXPECT_EQ(logged_cert.Hash(), mapping.Entry().mapping(0).entry_hash());
  }

  unordered_map<string, LoggedEntry> new_logged_certs;
  for (int i(0); i < 2; ++i) {
    LoggedEntry c;
    this->test_signer_.CreateUnique(&c);
    this->AddPendingEntry(&c);
    new_logged_certs.insert(make_pair(c.Hash(), c));
  }
  this->DeletePendingEntry(logged_cert);
  LOG(INFO) << "2";
  EXPECT_OK(this->tree_signer_->SequenceNewEntries());

  {
    EntryHandle<SequenceMapping> mapping;
    CHECK_EQ(Status::OK, this->store_->GetSequenceMapping(&mapping));
    CHECK_GE(mapping.Entry().mapping_size(), 0);
    EXPECT_EQ(new_logged_certs.size(),
              static_cast<size_t>(mapping.Entry().mapping_size()));
    for (int i(0); i < mapping.Entry().mapping_size(); ++i) {
      const auto& m(mapping.Entry().mapping(i));
      EXPECT_NE(new_logged_certs.end(), new_logged_certs.find(m.entry_hash()));
    }
  }
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
