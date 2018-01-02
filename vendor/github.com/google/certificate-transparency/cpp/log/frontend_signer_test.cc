/* -*- indent-tabs-mode: nil -*- */
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "log/etcd_consistent_store.h"
#include "log/file_db.h"
#include "log/frontend_signer.h"
#include "log/log_verifier.h"
#include "log/logged_entry.h"
#include "log/sqlite_db.h"
#include "log/test_db.h"
#include "log/test_signer.h"
#include "merkletree/merkle_verifier.h"
#include "merkletree/serial_hasher.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/fake_etcd.h"
#include "util/libevent_wrapper.h"
#include "util/mock_masterelection.h"
#include "util/status.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/thread_pool.h"
#include "util/util.h"

namespace {

namespace libevent = cert_trans::libevent;

using cert_trans::ConsistentStore;
using cert_trans::Database;
using cert_trans::EntryHandle;
using cert_trans::EtcdConsistentStore;
using cert_trans::FakeEtcdClient;
using cert_trans::FileDB;
using cert_trans::LoggedEntry;
using cert_trans::MockMasterElection;
using cert_trans::SQLiteDB;
using cert_trans::ThreadPool;
using ct::LogEntry;
using ct::SignedCertificateTimestamp;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using testing::NiceMock;
using testing::_;
using util::testing::StatusIs;

typedef FrontendSigner FS;

template <class T>
class FrontendSignerTest : public ::testing::Test {
 protected:
  FrontendSignerTest()
      : test_db_(),
        test_signer_(),
        verifier_(TestSigner::DefaultLogSigVerifier(),
                  new MerkleVerifier(new Sha256Hasher())),
        base_(make_shared<libevent::Base>()),
        event_pump_(base_),
        etcd_client_(base_.get()),
        pool_(2),
        store_(base_.get(), &pool_, &etcd_client_, &election_, "/root", "id"),
        log_signer_(TestSigner::DefaultLogSigner()),
        frontend_(db(), &store_, log_signer_.get()) {
  }

  T* db() const {
    return test_db_.db();
  }

  TestDB<T> test_db_;
  TestSigner test_signer_;
  LogVerifier verifier_;

  shared_ptr<libevent::Base> base_;
  libevent::EventPumpThread event_pump_;
  FakeEtcdClient etcd_client_;
  ThreadPool pool_;
  NiceMock<MockMasterElection> election_;
  EtcdConsistentStore<LoggedEntry> store_;
  unique_ptr<LogSigner> log_signer_;
  FS frontend_;
};

typedef testing::Types<FileDB, SQLiteDB> Databases;

TYPED_TEST_CASE(FrontendSignerTest, Databases);

TYPED_TEST(FrontendSignerTest, LogKatTest) {
  LogEntry default_entry;
  this->test_signer_.SetDefaults(&default_entry);

  // Log and expect success.
  EXPECT_OK(this->frontend_.QueueEntry(default_entry, NULL));

  // Look it up and expect to get the right thing back.
  string hash =
      Sha256Hasher::Sha256Digest(Serializer::LeafData(default_entry));
  EntryHandle<LoggedEntry> entry_handle;
  EXPECT_TRUE(this->store_.GetPendingEntryForHash(hash, &entry_handle).ok());
  const LoggedEntry& logged_cert(entry_handle.Entry());

  TestSigner::TestEqualEntries(default_entry, logged_cert.entry());
}

TYPED_TEST(FrontendSignerTest, Log) {
  LogEntry entry0, entry1;
  this->test_signer_.CreateUnique(&entry0);
  this->test_signer_.CreateUnique(&entry1);

  // Log and expect success.
  EXPECT_OK(this->frontend_.QueueEntry(entry0, NULL));
  EXPECT_OK(this->frontend_.QueueEntry(entry1, NULL));

  // Look it up and expect to get the right thing back.
  string hash0 = Sha256Hasher::Sha256Digest(Serializer::LeafData(entry0));
  string hash1 = Sha256Hasher::Sha256Digest(Serializer::LeafData(entry1));

  EntryHandle<LoggedEntry> entry_handle0;
  EntryHandle<LoggedEntry> entry_handle1;
  EXPECT_TRUE(this->store_.GetPendingEntryForHash(hash0, &entry_handle0).ok());
  EXPECT_TRUE(this->store_.GetPendingEntryForHash(hash1, &entry_handle1).ok());
  const LoggedEntry& logged_cert0(entry_handle0.Entry());
  const LoggedEntry& logged_cert1(entry_handle1.Entry());

  TestSigner::TestEqualEntries(entry0, logged_cert0.entry());
  TestSigner::TestEqualEntries(entry1, logged_cert1.entry());
}

TYPED_TEST(FrontendSignerTest, Time) {
  LogEntry entry0, entry1;
  this->test_signer_.CreateUnique(&entry0);
  this->test_signer_.CreateUnique(&entry1);

  // Log and expect success.
  SignedCertificateTimestamp sct0, sct1;
  EXPECT_OK(this->frontend_.QueueEntry(entry0, &sct0));
  EXPECT_LE(sct0.timestamp(), util::TimeInMilliseconds());
  EXPECT_GT(sct0.timestamp(), 0U);

  EXPECT_OK(this->frontend_.QueueEntry(entry1, &sct1));
  EXPECT_LE(sct0.timestamp(), sct1.timestamp());
  EXPECT_LE(sct1.timestamp(), util::TimeInMilliseconds());
}

TYPED_TEST(FrontendSignerTest, LogDuplicates) {
  LogEntry entry;
  this->test_signer_.CreateUnique(&entry);

  SignedCertificateTimestamp sct0, sct1;
  // Log and expect success.
  EXPECT_OK(this->frontend_.QueueEntry(entry, &sct0));
  // Wait for time to change.
  usleep(2000);
  // Try to log again.
  EXPECT_THAT(this->frontend_.QueueEntry(entry, &sct1),
              StatusIs(util::error::ALREADY_EXISTS, _));

  // Expect to get the original timestamp.
  EXPECT_EQ(sct0.timestamp(), sct1.timestamp());
}

TYPED_TEST(FrontendSignerTest, LogDuplicatesDifferentChain) {
  LogEntry entry0, entry1;
  this->test_signer_.CreateUnique(&entry0);
  entry1.CopyFrom(entry0);
  if (entry1.type() == ct::X509_ENTRY) {
    entry1.mutable_x509_entry()->add_certificate_chain(
        this->test_signer_.UniqueFakeCertBytestring());
  } else {
    CHECK_EQ(ct::PRECERT_ENTRY, entry1.type());
    entry1.mutable_precert_entry()->add_precertificate_chain(
        this->test_signer_.UniqueFakeCertBytestring());
  }

  SignedCertificateTimestamp sct0, sct1;
  // Log and expect success.
  EXPECT_OK(this->frontend_.QueueEntry(entry0, &sct0));
  // Wait for time to change.
  usleep(2000);
  // Try to log again.
  EXPECT_THAT(this->frontend_.QueueEntry(entry1, &sct1),
              StatusIs(util::error::ALREADY_EXISTS, _));

  // Expect to get the original timestamp.
  EXPECT_EQ(sct0.timestamp(), sct1.timestamp());
}

TYPED_TEST(FrontendSignerTest, Verify) {
  LogEntry entry0, entry1;
  this->test_signer_.CreateUnique(&entry0);
  this->test_signer_.CreateUnique(&entry1);

  // Log and expect success.
  SignedCertificateTimestamp sct0, sct1;
  EXPECT_OK(this->frontend_.QueueEntry(entry0, &sct0));
  EXPECT_OK(this->frontend_.QueueEntry(entry1, &sct1));

  // Verify results.

  EXPECT_EQ(this->verifier_.VerifySignedCertificateTimestamp(entry0, sct0),
            LogVerifier::VERIFY_OK);
  EXPECT_EQ(this->verifier_.VerifySignedCertificateTimestamp(entry1, sct1),
            LogVerifier::VERIFY_OK);

  // Swap the data and expect failure.
  EXPECT_EQ(this->verifier_.VerifySignedCertificateTimestamp(entry0, sct1),
            LogVerifier::INVALID_SIGNATURE);
}

TYPED_TEST(FrontendSignerTest, TimedVerify) {
  LogEntry entry0, entry1;
  this->test_signer_.CreateUnique(&entry0);
  this->test_signer_.CreateUnique(&entry1);

  uint64_t past_time = util::TimeInMilliseconds();
  usleep(2000);

  // Log and expect success.
  SignedCertificateTimestamp sct0, sct1;
  EXPECT_OK(this->frontend_.QueueEntry(entry0, &sct0));
  // Make sure we get different timestamps.
  usleep(2000);
  EXPECT_OK(this->frontend_.QueueEntry(entry1, &sct1));

  EXPECT_GT(sct1.timestamp(), sct0.timestamp());

  // Verify.
  EXPECT_EQ(this->verifier_.VerifySignedCertificateTimestamp(entry0, sct0),
            LogVerifier::VERIFY_OK);
  EXPECT_EQ(this->verifier_.VerifySignedCertificateTimestamp(entry1, sct1),
            LogVerifier::VERIFY_OK);

  // Go back to the past and expect verification to fail (since the sct is
  // from the future).
  EXPECT_EQ(this->verifier_.VerifySignedCertificateTimestamp(entry0, sct0, 0,
                                                             past_time),
            LogVerifier::INVALID_TIMESTAMP);

  // Swap timestamps and expect failure.
  SignedCertificateTimestamp wrong_sct(sct0);
  wrong_sct.set_timestamp(sct1.timestamp());
  EXPECT_EQ(this->verifier_.VerifySignedCertificateTimestamp(entry0,
                                                             wrong_sct),
            LogVerifier::INVALID_SIGNATURE);
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
