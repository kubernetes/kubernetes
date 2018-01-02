#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <string>

#include "log/cert_checker.h"
#include "log/cert_submission_handler.h"
#include "log/ct_extensions.h"
#include "proto/ct.pb.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/util.h"

// Valid certificates.
// Self-signed
static const char kCaCert[] = "ca-cert.pem";
// Issued by ca.pem
static const char kLeafCert[] = "test-cert.pem";
// Issued by ca.pem
static const char kCaPreCert[] = "ca-pre-cert.pem";
// Issued by ca-cert.pem
static const char kPreCert[] = "test-embedded-pre-cert.pem";
// Issued by ca-pre-cert.pem
static const char kPreWithPreCaCert[] =
    "test-embedded-with-preca-pre-cert.pem";
// Issued by ca-cert.pem
static const char kIntermediateCert[] = "intermediate-cert.pem";
// Issued by intermediate-cert.pem
static const char kChainLeafCert[] = "test-intermediate-cert.pem";

namespace {

using cert_trans::Cert;
using cert_trans::CertChain;
using cert_trans::CertChecker;
using cert_trans::CertSubmissionHandler;
using cert_trans::PreCertChain;
using ct::LogEntry;
using std::string;
using util::testing::StatusIs;

class CertSubmissionHandlerTest : public ::testing::Test {
 protected:
  string ca_;
  string leaf_;
  string ca_precert_;
  string precert_;
  string precert_with_preca_;
  string intermediate_;
  string chain_leaf_;
  const string cert_dir_;
  CertSubmissionHandler* handler_;
  CertChecker* checker_;

  CertSubmissionHandlerTest()
      : cert_dir_(FLAGS_test_srcdir + "/test/testdata"), handler_(NULL) {
  }

  void SetUp() {
    checker_ = new CertChecker();
    checker_->LoadTrustedCertificates(cert_dir_ + "/" + kCaCert);
    handler_ = new CertSubmissionHandler(checker_);
    CHECK(util::ReadBinaryFile(cert_dir_ + "/" + kCaCert, &ca_))
        << "Could not read test data from " << cert_dir_
        << ". Wrong --test_srcdir?";
    CHECK(util::ReadBinaryFile(cert_dir_ + "/" + kLeafCert, &leaf_));
    CHECK(util::ReadBinaryFile(cert_dir_ + "/" + kCaPreCert, &ca_precert_));
    CHECK(util::ReadBinaryFile(cert_dir_ + "/" + kPreCert, &precert_));
    CHECK(util::ReadBinaryFile(cert_dir_ + "/" + kPreWithPreCaCert,
                               &precert_with_preca_));
    CHECK(util::ReadBinaryFile(cert_dir_ + "/" + kIntermediateCert,
                               &intermediate_));
    CHECK(
        util::ReadBinaryFile(cert_dir_ + "/" + kChainLeafCert, &chain_leaf_));
  }

  ~CertSubmissionHandlerTest() {
    delete checker_;
    delete handler_;
  }
};

TEST_F(CertSubmissionHandlerTest, SubmitCert) {
  CertChain submission(leaf_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  // Submit a leaf cert.
  EXPECT_OK(handler_->ProcessX509Submission(&submission, &entry));
  EXPECT_TRUE(entry.has_x509_entry());
  EXPECT_FALSE(entry.has_precert_entry());
  EXPECT_TRUE(entry.x509_entry().has_leaf_certificate());
  // Chain should include the root.
  EXPECT_EQ(1, entry.x509_entry().certificate_chain_size());
}

TEST_F(CertSubmissionHandlerTest, SubmitEmptyCert) {
  CertChain submission("");
  EXPECT_FALSE(submission.IsLoaded());

  LogEntry entry;
  EXPECT_THAT(handler_->ProcessX509Submission(&submission, &entry),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertSubmissionHandlerTest, SubmitInvalidCert) {
  CertChain submission(
      "-----BEGIN CERTIFICATE-----\n"
      "invalid\n"
      "-----END CERTIFICATE-----");
  EXPECT_FALSE(submission.IsLoaded());

  LogEntry entry;
  EXPECT_THAT(handler_->ProcessX509Submission(&submission, &entry),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertSubmissionHandlerTest, SubmitChain) {
  // Submit a chain.
  CertChain submission(chain_leaf_ + intermediate_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  EXPECT_OK(handler_->ProcessX509Submission(&submission, &entry));
  EXPECT_TRUE(entry.x509_entry().has_leaf_certificate());
  EXPECT_EQ(2, entry.x509_entry().certificate_chain_size());
}

TEST_F(CertSubmissionHandlerTest, SubmitPartialChain) {
  CertChain submission(chain_leaf_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  // Submit a leaf cert with a missing intermediate.
  EXPECT_THAT(handler_->ProcessX509Submission(&submission, &entry),
              StatusIs(util::error::FAILED_PRECONDITION));
}

TEST_F(CertSubmissionHandlerTest, SubmitInvalidChain) {
  CertChain submission(leaf_ + leaf_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  // An invalid chain with two certs in wrong order.
  EXPECT_THAT(handler_->ProcessX509Submission(&submission, &entry),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertSubmissionHandlerTest, SubmitCertAsPreCert) {
  PreCertChain submission(leaf_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  // Various things are wrong here, so do not expect a specific error.
  EXPECT_FALSE(handler_->ProcessPreCertSubmission(&submission, &entry).ok());
}

TEST_F(CertSubmissionHandlerTest, SubmitCertChainAsPreCert) {
  PreCertChain submission(chain_leaf_ + intermediate_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  EXPECT_FALSE(handler_->ProcessPreCertSubmission(&submission, &entry).ok());
}

TEST_F(CertSubmissionHandlerTest, SubmitPreCertChain) {
  PreCertChain submission(precert_ + ca_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  EXPECT_OK(handler_->ProcessPreCertSubmission(&submission, &entry));
  EXPECT_TRUE(entry.has_precert_entry());
  EXPECT_FALSE(entry.has_x509_entry());
  EXPECT_TRUE(entry.precert_entry().has_pre_certificate());
  EXPECT_TRUE(entry.precert_entry().pre_cert().has_issuer_key_hash());
  EXPECT_TRUE(entry.precert_entry().pre_cert().has_tbs_certificate());

  // CA cert
  EXPECT_EQ(1, entry.precert_entry().precertificate_chain_size());
}

TEST_F(CertSubmissionHandlerTest, SubmitPreCertChainUsingPreCA) {
  PreCertChain submission(precert_with_preca_ + ca_precert_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  EXPECT_OK(handler_->ProcessPreCertSubmission(&submission, &entry));
  EXPECT_TRUE(entry.has_precert_entry());
  EXPECT_FALSE(entry.has_x509_entry());
  EXPECT_TRUE(entry.precert_entry().has_pre_certificate());
  EXPECT_TRUE(entry.precert_entry().pre_cert().has_issuer_key_hash());
  EXPECT_TRUE(entry.precert_entry().pre_cert().has_tbs_certificate());

  // Precert Signing Certificate + CA cert
  EXPECT_EQ(2, entry.precert_entry().precertificate_chain_size());
}

TEST_F(CertSubmissionHandlerTest, SubmitInvalidPreCertChain) {
  // Missing issuer.
  PreCertChain submission(precert_with_preca_);
  EXPECT_TRUE(submission.IsLoaded());

  LogEntry entry;
  EXPECT_FALSE(handler_->ProcessPreCertSubmission(&submission, &entry).ok());
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  OpenSSL_add_all_algorithms();
  ERR_load_crypto_strings();
  cert_trans::LoadCtExtensions();
  return RUN_ALL_TESTS();
}
