/* -*- indent-tabs-mode: nil -*- */
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "log/etcd_consistent_store.h"
#include "log/file_db.h"
#include "log/file_storage.h"
#include "log/log_lookup.h"
#include "log/log_signer.h"
#include "log/log_verifier.h"
#include "log/logged_entry.h"
#include "log/sqlite_db.h"
#include "log/test_db.h"
#include "log/test_signer.h"
#include "log/tree_signer.h"
#include "merkletree/merkle_verifier.h"
#include "merkletree/serial_hasher.h"
#include "proto/cert_serializer.h"
#include "util/fake_etcd.h"
#include "util/mock_masterelection.h"
#include "util/sync_task.h"
#include "util/testing.h"
#include "util/thread_pool.h"
#include "util/util.h"

namespace {

namespace libevent = cert_trans::libevent;

using cert_trans::Database;
using cert_trans::EntryHandle;
using cert_trans::EtcdClient;
using cert_trans::FakeEtcdClient;
using cert_trans::FileDB;
using cert_trans::LogLookup;
using cert_trans::LoggedEntry;
using cert_trans::MockMasterElection;
using cert_trans::SQLiteDB;
using cert_trans::ThreadPool;
using cert_trans::TreeSigner;
using ct::MerkleAuditProof;
using ct::SequenceMapping;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using testing::NiceMock;

typedef TreeSigner<LoggedEntry> TS;


template <class T>
class LogLookupTest : public ::testing::Test {
 protected:
  LogLookupTest()
      : test_db_(),
        base_(make_shared<libevent::Base>()),
        event_pump_(base_),
        etcd_client_(base_.get()),
        pool_(2),
        store_(base_.get(), &pool_, &etcd_client_, &election_, "/root", "id"),
        test_signer_(),
        log_signer_(TestSigner::DefaultLogSigner()),
        tree_signer_(std::chrono::duration<double>(0), db(),
                     unique_ptr<CompactMerkleTree>(
                         new CompactMerkleTree(new Sha256Hasher)),
                     &store_, log_signer_.get()),
        verifier_(TestSigner::DefaultLogSigVerifier(),
                  new MerkleVerifier(new Sha256Hasher())) {
    // Set some noddy STH so that we can call UpdateTree on the Tree Signer.
    store_.SetServingSTH(ct::SignedTreeHead());
    // Force an empty sequence mapping file:
    {
      util::SyncTask task(&pool_);
      EtcdClient::Response r;
      etcd_client_.ForceSet("/root/sequence_mapping", "", &r, task.task());
      task.Wait();
    }
  }


  void CreateSequencedEntry(LoggedEntry* logged_cert, int64_t seq) {
    CHECK_NOTNULL(logged_cert);
    CHECK_GE(seq, 0);
    logged_cert->clear_sequence_number();

    CHECK(this->store_.AddPendingEntry(logged_cert).ok());

    EntryHandle<SequenceMapping> mapping;
    CHECK(this->store_.GetSequenceMapping(&mapping).ok());
    SequenceMapping::Mapping* m(mapping.MutableEntry()->add_mapping());
    m->set_sequence_number(seq);
    m->set_entry_hash(logged_cert->Hash());
    CHECK(this->store_.UpdateSequenceMapping(&mapping).ok());
  }

  void UpdateTree() {
    // first need to populate the local DB with the sequenced entries in etcd:
    EntryHandle<SequenceMapping> mapping;
    CHECK(this->store_.GetSequenceMapping(&mapping).ok());

    for (const auto& m : mapping.Entry().mapping()) {
      EntryHandle<LoggedEntry> entry;
      CHECK_EQ(util::Status::OK,
               this->store_.GetPendingEntryForHash(m.entry_hash(), &entry));
      entry.MutableEntry()->set_sequence_number(m.sequence_number());
      CHECK_EQ(this->db()->OK,
               this->db()->CreateSequencedEntry(entry.Entry()));
    }

    // then do the actual update.
    EXPECT_EQ(TS::OK, this->tree_signer_.UpdateTree());
    this->db()->WriteTreeHead(this->tree_signer_.LatestSTH());
  }

  T* db() const {
    return test_db_.db();
  }


  TestDB<T> test_db_;
  shared_ptr<libevent::Base> base_;
  libevent::EventPumpThread event_pump_;
  FakeEtcdClient etcd_client_;
  ThreadPool pool_;
  NiceMock<MockMasterElection> election_;
  cert_trans::EtcdConsistentStore<LoggedEntry> store_;
  TestSigner test_signer_;
  unique_ptr<LogSigner> log_signer_;
  TS tree_signer_;
  LogVerifier verifier_;
};


typedef testing::Types<FileDB, SQLiteDB> Databases;

TYPED_TEST_CASE(LogLookupTest, Databases);


TYPED_TEST(LogLookupTest, Lookup) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->CreateSequencedEntry(&logged_cert, 0);

  MerkleAuditProof proof;
  this->UpdateTree();

  LogLookup lookup(this->db());
  // Look the new entry up.
  EXPECT_EQ(LogLookup::OK,
            lookup.AuditProof(logged_cert.merkle_leaf_hash(), &proof));
}


TYPED_TEST(LogLookupTest, NotFound) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->CreateSequencedEntry(&logged_cert, 0);

  MerkleAuditProof proof;
  this->UpdateTree();

  LogLookup lookup(this->db());

  // Look up using a wrong hash.
  string hash = this->test_signer_.UniqueHash();
  EXPECT_EQ(LogLookup::NOT_FOUND, lookup.AuditProof(hash, &proof));
}


TYPED_TEST(LogLookupTest, Update) {
  LogLookup lookup(this->db());
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->CreateSequencedEntry(&logged_cert, 0);

  MerkleAuditProof proof;
  this->UpdateTree();

  // Look the new entry up.
  EXPECT_EQ(LogLookup::OK,
            lookup.AuditProof(logged_cert.merkle_leaf_hash(), &proof));
}


// Verify that the audit proof constructed is correct (assuming the signer
// operates correctly). TODO(ekasper): KAT tests.
TYPED_TEST(LogLookupTest, Verify) {
  LoggedEntry logged_cert;
  this->test_signer_.CreateUnique(&logged_cert);
  this->CreateSequencedEntry(&logged_cert, 0);

  MerkleAuditProof proof;
  this->UpdateTree();

  LogLookup lookup(this->db());
  // Look the new entry up.
  EXPECT_EQ(LogLookup::OK,
            lookup.AuditProof(logged_cert.merkle_leaf_hash(), &proof));
  EXPECT_EQ(LogVerifier::VERIFY_OK,
            this->verifier_.VerifyMerkleAuditProof(logged_cert.entry(),
                                                   logged_cert.sct(), proof));
}


// Build a bigger tree so that we actually verify a non-empty path.
TYPED_TEST(LogLookupTest, VerifyWithPath) {
  LoggedEntry logged_certs[13];

  // Make the tree not balanced for extra fun.
  for (int i = 0; i < 13; ++i) {
    this->test_signer_.CreateUnique(&logged_certs[i]);
    this->CreateSequencedEntry(&logged_certs[i], i);
  }

  this->UpdateTree();

  LogLookup lookup(this->db());
  MerkleAuditProof proof;

  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(LogLookup::OK,
              lookup.AuditProof(logged_certs[i].merkle_leaf_hash(), &proof));
    EXPECT_EQ(LogVerifier::VERIFY_OK,
              this->verifier_.VerifyMerkleAuditProof(logged_certs[i].entry(),
                                                     logged_certs[i].sct(),
                                                     proof));
  }
}


}  // namespace


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
