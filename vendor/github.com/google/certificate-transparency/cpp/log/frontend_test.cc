/* -*- indent-tabs-mode: nil -*- */
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <memory>
#include <string>

#include "log/cert_submission_handler.h"
#include "log/ct_extensions.h"
#include "log/etcd_consistent_store.h"
#include "log/file_db.h"
#include "log/frontend.h"
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
#include "util/fake_etcd.h"
#include "util/libevent_wrapper.h"
#include "util/mock_masterelection.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/thread_pool.h"
#include "util/util.h"

//  Valid certificates.
// Self-signed
static const char kCaCert[] = "ca-cert.pem";
// Issued by ca-cert.pem
static const char kLeafCert[] = "test-cert.pem";
// Issued by ca.pem
static const char kCaPreCert[] = "ca-pre-cert.pem";
// Issued by ca-cert.pem
static const char kPreCert[] = "test-embedded-pre-cert.pem";
// Issued by ca-pre-cert.pem
static const char kPreWithPreCaCert[] =
    "test-embedded-with-preca-pre-cert.pem";
// The resulting embedded certs, issued by ca-cert.pem
static const char kEmbeddedCert[] = "test-embedded-cert.pem";
static const char kEmbeddedWithPreCaCert[] =
    "test-embedded-with-preca-cert.pem";
// Issued by ca-cert.pem
static const char kIntermediateCert[] = "intermediate-cert.pem";
// Issued by intermediate-cert.pem
static const char kChainLeafCert[] = "test-intermediate-cert.pem";

namespace {

namespace libevent = cert_trans::libevent;

using cert_trans::Cert;
using cert_trans::CertChain;
using cert_trans::CertChecker;
using cert_trans::CertSubmissionHandler;
using cert_trans::Database;
using cert_trans::EntryHandle;
using cert_trans::EtcdConsistentStore;
using cert_trans::FakeEtcdClient;
using cert_trans::FileDB;
using cert_trans::LoggedEntry;
using cert_trans::MockMasterElection;
using cert_trans::PreCertChain;
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

typedef Frontend FE;

// A slightly shorter notation for constructing hex strings from binary blobs.
string H(const string& byte_string) {
  return util::HexString(byte_string);
}

template <class T>
class FrontendTest : public ::testing::Test {
 protected:
  FrontendTest()
      : test_db_(),
        test_signer_(),
        verifier_(TestSigner::DefaultLogSigVerifier(),
                  new MerkleVerifier(new Sha256Hasher())),
        checker_(),
        base_(make_shared<libevent::Base>()),
        event_pump_(base_),
        etcd_client_(base_.get()),
        pool_(2),
        store_(base_.get(), &pool_, &etcd_client_, &election_, "/root", "id"),
        log_signer_(TestSigner::DefaultLogSigner()),
        submission_handler_(&checker_),
        frontend_(new FrontendSigner(db(), &store_, log_signer_.get())),
        cert_dir_(FLAGS_test_srcdir + "/test/testdata") {
  }

  void SetUp() {
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kLeafCert, &leaf_pem_))
        << "Could not read test data from " << cert_dir_
        << ". Wrong --test_srcdir?";
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kCaPreCert, &ca_precert_pem_));
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kPreCert, &precert_pem_));
    CHECK(util::ReadBinaryFile(cert_dir_ + "/" + kPreWithPreCaCert,
                               &precert_with_preca_pem_));
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kIntermediateCert,
                             &intermediate_pem_));
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kChainLeafCert,
                             &chain_leaf_pem_));
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kCaCert, &ca_pem_));
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kEmbeddedCert, &embedded_pem_));
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kEmbeddedWithPreCaCert,
                             &embedded_with_preca_pem_));
    CHECK(checker_.LoadTrustedCertificates(cert_dir_ + "/" + kCaCert));
  }


  T* db() const {
    return test_db_.db();
  }

  TestDB<T> test_db_;
  TestSigner test_signer_;
  LogVerifier verifier_;
  CertChecker checker_;
  shared_ptr<libevent::Base> base_;
  libevent::EventPumpThread event_pump_;
  FakeEtcdClient etcd_client_;
  ThreadPool pool_;
  NiceMock<MockMasterElection> election_;
  EtcdConsistentStore<LoggedEntry> store_;
  unique_ptr<LogSigner> log_signer_;
  CertSubmissionHandler submission_handler_;
  FE frontend_;
  const string cert_dir_;
  string leaf_pem_;
  string ca_precert_pem_;
  string precert_pem_;
  string precert_with_preca_pem_;
  string intermediate_pem_;
  string chain_leaf_pem_;
  string embedded_pem_;
  string embedded_with_preca_pem_;
  string ca_pem_;
};

typedef testing::Types<FileDB, SQLiteDB> Databases;

TYPED_TEST_CASE(FrontendTest, Databases);

TYPED_TEST(FrontendTest, TestSubmitValid) {
  CertChain chain(this->leaf_pem_);
  EXPECT_TRUE(chain.IsLoaded());

  LogEntry entry;
  SignedCertificateTimestamp sct;
  EXPECT_OK(this->frontend_.QueueProcessedEntry(
      this->submission_handler_.ProcessX509Submission(&chain, &entry), entry,
      &sct));

  // Look it up and expect to get the right thing back.
  EntryHandle<LoggedEntry> entry_handle;
  Cert cert(this->leaf_pem_);

  string sha256_digest;
  ASSERT_OK(cert.Sha256Digest(&sha256_digest));
  EXPECT_TRUE(
      this->store_.GetPendingEntryForHash(sha256_digest, &entry_handle).ok());
  const LoggedEntry& logged_cert(entry_handle.Entry());

  EXPECT_EQ(ct::X509_ENTRY, logged_cert.entry().type());
  // Compare the leaf cert.
  string der_string;
  ASSERT_OK(cert.DerEncoding(&der_string));
  EXPECT_EQ(H(der_string),
            H(logged_cert.entry().x509_entry().leaf_certificate()));

  // And verify the signature.
  EXPECT_EQ(LogVerifier::VERIFY_OK,
            this->verifier_.VerifySignedCertificateTimestamp(
                logged_cert.entry(), sct));
}

TYPED_TEST(FrontendTest, TestSubmitValidWithIntermediate) {
  CertChain chain(this->chain_leaf_pem_ + this->intermediate_pem_);
  EXPECT_TRUE(chain.IsLoaded());

  LogEntry entry;
  SignedCertificateTimestamp sct;
  EXPECT_OK(this->frontend_.QueueProcessedEntry(
      this->submission_handler_.ProcessX509Submission(&chain, &entry), entry,
      &sct));

  // Look it up and expect to get the right thing back.
  Cert cert(this->chain_leaf_pem_);

  string sha256_digest;
  ASSERT_OK(cert.Sha256Digest(&sha256_digest));
  EntryHandle<LoggedEntry> entry_handle;
  EXPECT_TRUE(
      this->store_.GetPendingEntryForHash(sha256_digest, &entry_handle).ok());
  const LoggedEntry& logged_cert(entry_handle.Entry());

  EXPECT_EQ(ct::X509_ENTRY, logged_cert.entry().type());
  // Compare the leaf cert.
  string der_string;
  ASSERT_OK(cert.DerEncoding(&der_string));
  EXPECT_EQ(H(der_string),
            H(logged_cert.entry().x509_entry().leaf_certificate()));

  // And verify the signature.
  EXPECT_EQ(LogVerifier::VERIFY_OK,
            this->verifier_.VerifySignedCertificateTimestamp(
                logged_cert.entry(), sct));

  // Compare the first intermediate.
  ASSERT_GE(logged_cert.entry().x509_entry().certificate_chain_size(), 1);
  Cert cert2(this->intermediate_pem_);

  ASSERT_OK(cert2.DerEncoding(&der_string));
  EXPECT_EQ(H(der_string),
            H(logged_cert.entry().x509_entry().certificate_chain(0)));
}

TYPED_TEST(FrontendTest, TestSubmitDuplicate) {
  CertChain chain1(this->leaf_pem_);
  CertChain chain2(this->leaf_pem_);
  EXPECT_TRUE(chain1.IsLoaded());
  EXPECT_TRUE(chain2.IsLoaded());

  SignedCertificateTimestamp sct;
  LogEntry entry1, entry2;
  EXPECT_OK(this->frontend_.QueueProcessedEntry(
      this->submission_handler_.ProcessX509Submission(&chain1, &entry1),
      entry1, nullptr));
  EXPECT_THAT(this->frontend_.QueueProcessedEntry(
                  this->submission_handler_.ProcessX509Submission(&chain2,
                                                                  &entry2),
                  entry2, &sct),
              StatusIs(util::error::ALREADY_EXISTS, _));

  // Look it up and expect to get the right thing back.
  Cert cert(this->leaf_pem_);

  string sha256_digest;
  ASSERT_OK(cert.Sha256Digest(&sha256_digest));
  EntryHandle<LoggedEntry> entry_handle;
  EXPECT_TRUE(
      this->store_.GetPendingEntryForHash(sha256_digest, &entry_handle).ok());
  const LoggedEntry& logged_cert(entry_handle.Entry());

  EXPECT_EQ(ct::X509_ENTRY, logged_cert.entry().type());
  // Compare the leaf cert.
  string der_string;
  ASSERT_OK(cert.DerEncoding(&der_string));
  EXPECT_EQ(H(der_string),
            H(logged_cert.entry().x509_entry().leaf_certificate()));

  // And verify the signature.
  EXPECT_EQ(LogVerifier::VERIFY_OK,
            this->verifier_.VerifySignedCertificateTimestamp(
                logged_cert.entry(), sct));
}

TYPED_TEST(FrontendTest, TestSubmitInvalidChain) {
  CertChain chain(this->chain_leaf_pem_);
  EXPECT_TRUE(chain.IsLoaded());

  LogEntry entry;
  SignedCertificateTimestamp sct;
  // Missing intermediate.
  EXPECT_THAT(this->frontend_.QueueProcessedEntry(
                  this->submission_handler_.ProcessX509Submission(&chain,
                                                                  &entry),
                  entry, &sct),
              StatusIs(util::error::FAILED_PRECONDITION, "unknown root"));
  EXPECT_FALSE(sct.has_signature());
}

TYPED_TEST(FrontendTest, TestSubmitInvalidPem) {
  CertChain chain(
      "-----BEGIN CERTIFICATE-----\n"
      "Iamnotavalidcert\n"
      "-----END CERTIFICATE-----\n");
  EXPECT_FALSE(chain.IsLoaded());

  LogEntry entry;
  SignedCertificateTimestamp sct;
  EXPECT_THAT(this->frontend_.QueueProcessedEntry(
                  this->submission_handler_.ProcessX509Submission(&chain,
                                                                  &entry),
                  entry, &sct),
              StatusIs(util::error::INVALID_ARGUMENT, "empty submission"));
  EXPECT_FALSE(sct.has_signature());
}

TYPED_TEST(FrontendTest, TestSubmitPrecert) {
  PreCertChain submission(this->precert_pem_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry log_entry;
  SignedCertificateTimestamp sct;
  EXPECT_OK(this->frontend_.QueueProcessedEntry(
      this->submission_handler_.ProcessPreCertSubmission(&submission,
                                                         &log_entry),
      log_entry, &sct));

  CertChain chain(this->embedded_pem_ + this->ca_pem_);
  LogEntry entry;
  CertSubmissionHandler::X509ChainToEntry(chain, &entry);

  // Look it up.
  string hash = Sha256Hasher::Sha256Digest(
      entry.precert_entry().pre_cert().tbs_certificate());
  EntryHandle<LoggedEntry> entry_handle;
  EXPECT_TRUE(this->store_.GetPendingEntryForHash(hash, &entry_handle).ok());
  const LoggedEntry& logged_cert(entry_handle.Entry());
  Cert pre(this->precert_pem_);
  Cert ca(this->ca_pem_);

  EXPECT_EQ(ct::PRECERT_ENTRY, logged_cert.entry().type());
  // Verify the signature.
  EXPECT_EQ(LogVerifier::VERIFY_OK,
            this->verifier_.VerifySignedCertificateTimestamp(
                logged_cert.entry(), sct));

  // Expect to have the original certs logged in the chain.
  ASSERT_EQ(logged_cert.entry().precert_entry().precertificate_chain_size(),
            1);

  string pre_der, ca_der;
  ASSERT_OK(pre.DerEncoding(&pre_der));
  ASSERT_OK(ca.DerEncoding(&ca_der));

  EXPECT_EQ(H(pre_der),
            H(logged_cert.entry().precert_entry().pre_certificate()));
  EXPECT_EQ(H(ca_der),
            H(logged_cert.entry().precert_entry().precertificate_chain(0)));
}

TYPED_TEST(FrontendTest, TestSubmitPrecertUsingPreCA) {
  PreCertChain submission(this->precert_with_preca_pem_ +
                          this->ca_precert_pem_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry log_entry;
  SignedCertificateTimestamp sct;
  EXPECT_OK(this->frontend_.QueueProcessedEntry(
      this->submission_handler_.ProcessPreCertSubmission(&submission,
                                                         &log_entry),
      log_entry, &sct));

  CertChain chain(this->embedded_with_preca_pem_ + this->ca_pem_);
  LogEntry entry;
  CertSubmissionHandler::X509ChainToEntry(chain, &entry);

  // Look it up.
  string hash = Sha256Hasher::Sha256Digest(
      entry.precert_entry().pre_cert().tbs_certificate());
  EntryHandle<LoggedEntry> entry_handle;
  EXPECT_TRUE(this->store_.GetPendingEntryForHash(hash, &entry_handle).ok());
  const LoggedEntry& logged_cert(entry_handle.Entry());
  Cert pre(this->precert_with_preca_pem_);
  Cert ca_pre(this->ca_precert_pem_);
  Cert ca(this->ca_pem_);

  EXPECT_EQ(ct::PRECERT_ENTRY, logged_cert.entry().type());
  // Verify the signature.
  EXPECT_EQ(LogVerifier::VERIFY_OK,
            this->verifier_.VerifySignedCertificateTimestamp(
                logged_cert.entry(), sct));

  // Expect to have the original certs logged in the chain.
  ASSERT_GE(logged_cert.entry().precert_entry().precertificate_chain_size(),
            2);

  string pre_der, ca_der, ca_pre_der;
  ASSERT_OK(pre.DerEncoding(&pre_der));
  ASSERT_OK(ca.DerEncoding(&ca_der));
  ASSERT_OK(ca_pre.DerEncoding(&ca_pre_der));

  EXPECT_EQ(H(pre_der),
            H(logged_cert.entry().precert_entry().pre_certificate()));
  EXPECT_EQ(H(ca_pre_der),
            H(logged_cert.entry().precert_entry().precertificate_chain(0)));
  EXPECT_EQ(H(ca_der),
            H(logged_cert.entry().precert_entry().precertificate_chain(1)));
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  OpenSSL_add_all_algorithms();
  ERR_load_crypto_strings();
  cert_trans::LoadCtExtensions();
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
