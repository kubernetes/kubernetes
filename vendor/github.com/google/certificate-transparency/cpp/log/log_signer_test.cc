/* -*- indent-tabs-mode: nil -*- */
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <string>

#include "log/log_signer.h"
#include "log/test_signer.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/testing.h"
#include "util/util.h"

namespace {

using ct::LogEntry;
using ct::SignedCertificateTimestamp;
using ct::DigitallySigned;
using ct::SignedTreeHead;
using std::string;

// A slightly shorter notation for constructing hex strings from binary blobs.
string H(const string& byte_string) {
  return util::HexString(byte_string);
}

class LogSignerTest : public ::testing::Test {
 protected:
  LogSignerTest() : signer_(NULL), verifier_(NULL) {
  }

  void SetUp() {
    signer_ = TestSigner::DefaultLogSigner();
    verifier_ = TestSigner::DefaultLogSigVerifier();
  }

  ~LogSignerTest() {
    delete signer_;
    delete verifier_;
  }

  static string SerializedSignature(const DigitallySigned& signature) {
    string serialized_sig;
    CHECK_EQ(SerializeResult::OK,
             Serializer::SerializeDigitallySigned(signature, &serialized_sig));
    return serialized_sig;
  }

  LogSigner* signer_;
  LogSigVerifier* verifier_;
  TestSigner test_signer_;
};

TEST_F(LogSignerTest, KeyIDKatTest) {
  SignedCertificateTimestamp default_sct;
  TestSigner::SetDefaults(&default_sct);
  EXPECT_EQ(signer_->KeyID(), default_sct.id().key_id());
  EXPECT_EQ(verifier_->KeyID(), default_sct.id().key_id());
}

TEST_F(LogSignerTest, VerifyCertSCTKatTest) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);

  SignedCertificateTimestamp default_sct;
  TestSigner::SetDefaults(&default_sct);

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, default_sct));

  CHECK_EQ(default_entry.type(), ct::X509_ENTRY);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
}

TEST_F(LogSignerTest, VerifyPrecertSCTKatTest) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);

  SignedCertificateTimestamp default_sct;
  TestSigner::SetPrecertDefaults(&default_sct);

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, default_sct));

  CHECK_EQ(default_entry.type(), ct::PRECERT_ENTRY);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
}

TEST_F(LogSignerTest, VerifySTHKatTest) {
  SignedTreeHead default_sth;
  TestSigner::SetDefaults(&default_sth);

  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(default_sth));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1STHSignature(
                default_sth.timestamp(), default_sth.tree_size(),
                default_sth.sha256_root_hash(),
                SerializedSignature(default_sth.signature())));
}

TEST_F(LogSignerTest, SignAndVerifyCertSCT) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();
  ASSERT_FALSE(sct.has_signature());

  EXPECT_EQ(LogSigner::OK,
            signer_->SignCertificateTimestamp(default_entry, &sct));
  EXPECT_TRUE(sct.has_signature());
  EXPECT_EQ(default_sct.signature().hash_algorithm(),
            sct.signature().hash_algorithm());
  EXPECT_EQ(default_sct.signature().sig_algorithm(),
            sct.signature().sig_algorithm());
  // We should get a fresh signature.
  EXPECT_NE(H(default_sct.signature().signature()),
            H(sct.signature().signature()));
  // But it should still be valid.
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  // The second version.
  CHECK_EQ(default_entry.type(), ct::X509_ENTRY);
  string serialized_sig;
  EXPECT_EQ(LogSigner::OK, signer_->SignV1CertificateTimestamp(
                               default_sct.timestamp(),
                               default_entry.x509_entry().leaf_certificate(),
                               default_sct.extensions(), &serialized_sig));

  string default_serialized_sig = SerializedSignature(default_sct.signature());
  EXPECT_NE(H(default_serialized_sig), H(serialized_sig));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(), serialized_sig));
}

TEST_F(LogSignerTest, SignAndVerifyPrecertSCT) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();
  ASSERT_FALSE(sct.has_signature());

  EXPECT_EQ(LogSigner::OK,
            signer_->SignCertificateTimestamp(default_entry, &sct));
  EXPECT_TRUE(sct.has_signature());
  EXPECT_EQ(default_sct.signature().hash_algorithm(),
            sct.signature().hash_algorithm());
  EXPECT_EQ(default_sct.signature().sig_algorithm(),
            sct.signature().sig_algorithm());
  // We should get a fresh signature.
  EXPECT_NE(H(default_sct.signature().signature()),
            H(sct.signature().signature()));
  // But it should still be valid.
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  // The second version.
  CHECK_EQ(default_entry.type(), ct::PRECERT_ENTRY);
  string serialized_sig;
  EXPECT_EQ(LogSigner::OK,
            signer_->SignV1PrecertificateTimestamp(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(), &serialized_sig));

  string default_serialized_sig = SerializedSignature(default_sct.signature());
  EXPECT_NE(H(default_serialized_sig), H(serialized_sig));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(), serialized_sig));
}

TEST_F(LogSignerTest, SignAndVerifySTH) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  sth.CopyFrom(default_sth);
  sth.clear_signature();
  ASSERT_FALSE(sth.has_signature());

  EXPECT_EQ(LogSigner::OK, signer_->SignTreeHead(&sth));
  EXPECT_TRUE(sth.has_signature());
  EXPECT_EQ(default_sth.signature().hash_algorithm(),
            sth.signature().hash_algorithm());
  EXPECT_EQ(default_sth.signature().sig_algorithm(),
            sth.signature().sig_algorithm());
  // We should get a fresh signature.
  EXPECT_NE(H(default_sth.signature().signature()),
            H(sth.signature().signature()));
  // But it should still be valid.
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

  // The second version.
  string serialized_sig;
  EXPECT_EQ(LogSigner::OK,
            signer_->SignV1TreeHead(default_sth.timestamp(),
                                    default_sth.tree_size(),
                                    default_sth.sha256_root_hash(),
                                    &serialized_sig));

  string default_serialized_sig = SerializedSignature(default_sth.signature());
  EXPECT_NE(H(default_serialized_sig), H(serialized_sig));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1STHSignature(default_sth.timestamp(),
                                            default_sth.tree_size(),
                                            default_sth.sha256_root_hash(),
                                            default_serialized_sig));
}

TEST_F(LogSignerTest, SignAndVerifyCertSCTApiCrossCheck) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();

  EXPECT_EQ(LogSigner::OK,
            signer_->SignCertificateTimestamp(default_entry, &sct));

  // Serialize and verify.
  string serialized_sig;
  EXPECT_EQ(SerializeResult::OK,
            Serializer::SerializeDigitallySigned(sct.signature(),
                                                 &serialized_sig));

  CHECK_EQ(default_entry.type(), ct::X509_ENTRY);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(), serialized_sig));

  // The second version.
  serialized_sig.clear();
  EXPECT_EQ(LogSigner::OK, signer_->SignV1CertificateTimestamp(
                               default_sct.timestamp(),
                               default_entry.x509_entry().leaf_certificate(),
                               default_sct.extensions(), &serialized_sig));

  // Deserialize and verify.
  sct.clear_signature();
  EXPECT_EQ(DeserializeResult::OK,
            Deserializer::DeserializeDigitallySigned(serialized_sig,
                                                     sct.mutable_signature()));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));
}

TEST_F(LogSignerTest, SignAndVerifyPrecertSCTApiCrossCheck) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();

  EXPECT_EQ(LogSigner::OK,
            signer_->SignCertificateTimestamp(default_entry, &sct));

  // Serialize and verify.
  string serialized_sig;
  EXPECT_EQ(SerializeResult::OK,
            Serializer::SerializeDigitallySigned(sct.signature(),
                                                 &serialized_sig));

  CHECK_EQ(default_entry.type(), ct::PRECERT_ENTRY);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(), serialized_sig));

  // The second version.
  serialized_sig.clear();
  EXPECT_EQ(LogSigner::OK,
            signer_->SignV1PrecertificateTimestamp(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(), &serialized_sig));

  // Deserialize and verify.
  sct.clear_signature();
  EXPECT_EQ(DeserializeResult::OK,
            Deserializer::DeserializeDigitallySigned(serialized_sig,
                                                     sct.mutable_signature()));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));
}

TEST_F(LogSignerTest, SignAndVerifySTHApiCrossCheck) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  sth.CopyFrom(default_sth);
  sth.clear_signature();

  EXPECT_EQ(LogSigner::OK, signer_->SignTreeHead(&sth));

  // Serialize and verify.
  string serialized_sig;
  EXPECT_EQ(SerializeResult::OK,
            Serializer::SerializeDigitallySigned(sth.signature(),
                                                 &serialized_sig));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1STHSignature(default_sth.timestamp(),
                                            default_sth.tree_size(),
                                            default_sth.sha256_root_hash(),
                                            serialized_sig));

  // The second version.
  serialized_sig.clear();
  EXPECT_EQ(LogSigner::OK,
            signer_->SignV1TreeHead(default_sth.timestamp(),
                                    default_sth.tree_size(),
                                    default_sth.sha256_root_hash(),
                                    &serialized_sig));

  // Deserialize and verify.
  sth.clear_signature();
  EXPECT_EQ(DeserializeResult::OK,
            Deserializer::DeserializeDigitallySigned(serialized_sig,
                                                     sth.mutable_signature()));
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));
}

TEST_F(LogSignerTest, SignInvalidType) {
  LogEntry default_entry, entry;
  TestSigner::SetDefaults(&default_entry);

  entry.CopyFrom(default_entry);
  entry.set_type(ct::UNKNOWN_ENTRY_TYPE);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();

  string serialized_sig;
  EXPECT_EQ(LogSigner::INVALID_ENTRY_TYPE,
            signer_->SignCertificateTimestamp(entry, &sct));
}

TEST_F(LogSignerTest, SignEmptyCert) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);

  CHECK_EQ(ct::X509_ENTRY, default_entry.type());
  LogEntry entry;
  entry.CopyFrom(default_entry);
  entry.mutable_x509_entry()->clear_leaf_certificate();

  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();

  EXPECT_EQ(LogSigner::EMPTY_CERTIFICATE,
            signer_->SignCertificateTimestamp(entry, &sct));

  string serialized_sig;
  string empty_cert;
  EXPECT_EQ(LogSigner::EMPTY_CERTIFICATE,
            signer_->SignV1CertificateTimestamp(default_sct.timestamp(),
                                                empty_cert,
                                                default_sct.extensions(),
                                                &serialized_sig));
}

TEST_F(LogSignerTest, SignEmptyPreCert) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);

  CHECK_EQ(ct::PRECERT_ENTRY, default_entry.type());
  LogEntry entry;
  entry.CopyFrom(default_entry);
  entry.mutable_precert_entry()->mutable_pre_cert()->clear_tbs_certificate();

  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();

  EXPECT_EQ(LogSigner::EMPTY_CERTIFICATE,
            signer_->SignCertificateTimestamp(entry, &sct));

  string serialized_sig;
  string empty_cert;
  EXPECT_EQ(LogSigner::EMPTY_CERTIFICATE,
            signer_->SignV1PrecertificateTimestamp(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                empty_cert, default_sct.extensions(), &serialized_sig));
}

TEST_F(LogSignerTest, SignInvalidIssuerKeyHash) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);

  CHECK_EQ(ct::PRECERT_ENTRY, default_entry.type());
  LogEntry entry;
  entry.CopyFrom(default_entry);
  entry.mutable_precert_entry()->mutable_pre_cert()->clear_issuer_key_hash();

  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  sct.clear_signature();

  EXPECT_EQ(LogSigner::INVALID_HASH_LENGTH,
            signer_->SignCertificateTimestamp(entry, &sct));

  string serialized_sig;
  string bad_hash("too short");
  EXPECT_EQ(LogSigner::INVALID_HASH_LENGTH,
            signer_->SignV1PrecertificateTimestamp(
                default_sct.timestamp(), bad_hash,
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(), &serialized_sig));
}

TEST_F(LogSignerTest, SignBadRootHash) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  sth.CopyFrom(default_sth);
  sth.clear_signature();
  sth.set_sha256_root_hash("bad");

  EXPECT_EQ(LogSigner::INVALID_HASH_LENGTH, signer_->SignTreeHead(&sth));

  string serialized_sig;
  EXPECT_EQ(LogSigner::INVALID_HASH_LENGTH,
            signer_->SignV1TreeHead(default_sth.timestamp(),
                                    default_sth.tree_size(), "bad",
                                    &serialized_sig));
}

TEST_F(LogSignerTest, VerifyChangeCertSCTTimestamp) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  uint64_t new_timestamp = default_sct.timestamp() + 1000;

  sct.set_timestamp(new_timestamp);
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(default_entry, sct));

  CHECK_EQ(ct::X509_ENTRY, default_entry.type());
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));

  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1CertSCTSignature(
                new_timestamp, default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
}

TEST_F(LogSignerTest, VerifyChangePrecertSCTTimestamp) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  uint64_t new_timestamp = default_sct.timestamp() + 1000;

  sct.set_timestamp(new_timestamp);
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(default_entry, sct));

  CHECK_EQ(ct::PRECERT_ENTRY, default_entry.type());
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));

  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1PrecertSCTSignature(
                new_timestamp,
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
}

TEST_F(LogSignerTest, VerifyChangeSTHTimestamp) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  sth.CopyFrom(default_sth);
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

  uint64_t new_timestamp = default_sth.timestamp() + 1000;
  sth.set_timestamp(new_timestamp);
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySTHSignature(sth));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1STHSignature(
                default_sth.timestamp(), default_sth.tree_size(),
                default_sth.sha256_root_hash(),
                SerializedSignature(default_sth.signature())));

  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1STHSignature(
                new_timestamp, default_sth.tree_size(),
                default_sth.sha256_root_hash(),
                SerializedSignature(default_sth.signature())));
}

TEST_F(LogSignerTest, VerifyChangeCert) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct, sct2;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  CHECK_EQ(ct::X509_ENTRY, default_entry.type());
  LogEntry entry;
  entry.CopyFrom(default_entry);
  string new_cert = test_signer_.UniqueFakeCertBytestring();
  entry.mutable_x509_entry()->set_leaf_certificate(new_cert);

  // Check that we can successfully sign and verify the new sct.
  sct2.CopyFrom(sct);
  EXPECT_EQ(LogSigner::OK, signer_->SignCertificateTimestamp(entry, &sct2));
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySCTSignature(entry, sct2));

  // We should not be able to verify the new cert with the old signature.
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(entry, sct));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(), new_cert, default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
}

TEST_F(LogSignerTest, VerifyChangePrecert) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct, sct2;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  CHECK_EQ(ct::PRECERT_ENTRY, default_entry.type());
  LogEntry entry;
  entry.CopyFrom(default_entry);
  string new_cert = test_signer_.UniqueFakeCertBytestring();
  entry.mutable_precert_entry()->mutable_pre_cert()->set_tbs_certificate(
      new_cert);

  // Check that we can successfully sign and verify the new sct.
  sct2.CopyFrom(sct);
  EXPECT_EQ(LogSigner::OK, signer_->SignCertificateTimestamp(entry, &sct2));
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySCTSignature(entry, sct2));

  // We should not be able to verify the new cert with the old signature.
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(entry, sct));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                new_cert, default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
}

TEST_F(LogSignerTest, VerifyChangeIssuerKeyHash) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct, sct2;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  CHECK_EQ(ct::PRECERT_ENTRY, default_entry.type());
  LogEntry entry;
  entry.CopyFrom(default_entry);
  string new_hash = test_signer_.UniqueHash();
  entry.mutable_precert_entry()->mutable_pre_cert()->set_issuer_key_hash(
      new_hash);

  // Check that we can successfully sign and verify the new sct.
  sct2.CopyFrom(sct);
  EXPECT_EQ(LogSigner::OK, signer_->SignCertificateTimestamp(entry, &sct2));
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySCTSignature(entry, sct2));

  // We should not be able to verify the new cert with the old signature.
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(entry, sct));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1PrecertSCTSignature(
                default_sct.timestamp(), new_hash,
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                default_sct.extensions(),
                SerializedSignature(default_sct.signature())));
}

TEST_F(LogSignerTest, VerifyChangeCertExtensions) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct, sct2;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                sct.timestamp(), default_entry.x509_entry().leaf_certificate(),
                sct.extensions(), SerializedSignature(sct.signature())));

  sct.set_extensions("hello");
  // Check that we can successfully sign and verify the new sct.
  sct2.CopyFrom(sct);
  EXPECT_EQ(LogSigner::OK,
            signer_->SignCertificateTimestamp(default_entry, &sct2));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct2));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                sct2.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                sct2.extensions(), SerializedSignature(sct2.signature())));

  // We should not be able to verify the new data with the old signature.
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(default_entry, sct));

  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1CertSCTSignature(
                sct.timestamp(), default_entry.x509_entry().leaf_certificate(),
                sct.extensions(), SerializedSignature(sct.signature())));
}

TEST_F(LogSignerTest, VerifyChangePrecertExtensions) {
  LogEntry default_entry;
  TestSigner::SetPrecertDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct, sct2;
  TestSigner::SetPrecertDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                sct.extensions(), SerializedSignature(sct.signature())));

  sct.set_extensions("hello");
  // Check that we can successfully sign and verify the new sct.
  sct2.CopyFrom(sct);
  EXPECT_EQ(LogSigner::OK,
            signer_->SignCertificateTimestamp(default_entry, &sct2));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct2));
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1PrecertSCTSignature(
                sct2.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                sct2.extensions(), SerializedSignature(sct2.signature())));

  // We should not be able to verify the new data with the old signature.
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(default_entry, sct));

  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1PrecertSCTSignature(
                sct.timestamp(),
                default_entry.precert_entry().pre_cert().issuer_key_hash(),
                default_entry.precert_entry().pre_cert().tbs_certificate(),
                sct.extensions(), SerializedSignature(sct.signature())));
}

TEST_F(LogSignerTest, VerifyChangeTreeSize) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  sth.CopyFrom(default_sth);
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

  ASSERT_GE(default_sth.tree_size(), 0);
  int64_t new_tree_size = default_sth.tree_size() + 1;
  ASSERT_GE(new_tree_size, 0);
  sth.set_tree_size(new_tree_size);
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySTHSignature(sth));

  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1STHSignature(
                default_sth.timestamp(), default_sth.tree_size(),
                default_sth.sha256_root_hash(),
                SerializedSignature(default_sth.signature())));

  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifyV1STHSignature(
                default_sth.timestamp(), new_tree_size,
                default_sth.sha256_root_hash(),
                SerializedSignature(default_sth.signature())));
}

TEST_F(LogSignerTest, VerifySCTBadHashAlgorithm) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  CHECK_NE(DigitallySigned::SHA224, sct.signature().hash_algorithm());
  sct.mutable_signature()->set_hash_algorithm(DigitallySigned::SHA224);
  EXPECT_EQ(LogSigVerifier::HASH_ALGORITHM_MISMATCH,
            verifier_->VerifySCTSignature(default_entry, sct));
}

TEST_F(LogSignerTest, VerifySTHBadHashAlgorithm) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  sth.CopyFrom(default_sth);
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

  CHECK_NE(DigitallySigned::SHA224, sth.signature().hash_algorithm());
  sth.mutable_signature()->set_hash_algorithm(DigitallySigned::SHA224);
  EXPECT_EQ(LogSigVerifier::HASH_ALGORITHM_MISMATCH,
            verifier_->VerifySTHSignature(sth));
}

TEST_F(LogSignerTest, VerifySCTBadSignatureAlgorithm) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  CHECK_NE(DigitallySigned::DSA, sct.signature().sig_algorithm());
  sct.mutable_signature()->set_sig_algorithm(DigitallySigned::DSA);
  EXPECT_EQ(LogSigVerifier::SIGNATURE_ALGORITHM_MISMATCH,
            verifier_->VerifySCTSignature(default_entry, sct));
}

TEST_F(LogSignerTest, VerifySTHBadSignatureAlgorithm) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  sth.CopyFrom(default_sth);
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

  CHECK_NE(DigitallySigned::DSA, sth.signature().sig_algorithm());
  sth.mutable_signature()->set_sig_algorithm(DigitallySigned::DSA);
  EXPECT_EQ(LogSigVerifier::SIGNATURE_ALGORITHM_MISMATCH,
            verifier_->VerifySTHSignature(sth));
}

TEST_F(LogSignerTest, VerifyBadSCTSignature) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  // Too short.
  sct.CopyFrom(default_sct);
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifySCTSignature(default_entry, sct));

  string bad_signature = default_sct.signature().signature();
  bad_signature.erase(bad_signature.end() - 1);
  sct.mutable_signature()->set_signature(bad_signature);
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySCTSignature(default_entry, sct));

  // Too long.
  // OpenSSL ECDSA Verify parses *up to* a given number of bytes,
  // rather than exactly the given number of bytes, and hence appending
  // garbage in the end still results in a valid signature.
  // sct.CopyFrom(default_sct);
  // EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySCTSignature(sct));

  // bad_signature = default_sct.signature().signature();
  // bad_signature.push_back(0x42);

  // sct.mutable_signature()->set_signature(bad_signature);
  // EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
  // verifier_->VerifySCTSignature(sct));

  // Flip the lsb of each byte one by one.
  for (size_t i = 0; i < default_sct.signature().signature().size(); ++i) {
    sct.CopyFrom(default_sct);
    EXPECT_EQ(LogSigVerifier::OK,
              verifier_->VerifySCTSignature(default_entry, sct));

    bad_signature = default_sct.signature().signature();
    bad_signature[i] ^= 0x01;
    sct.mutable_signature()->set_signature(bad_signature);
    EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
              verifier_->VerifySCTSignature(default_entry, sct));
  }
}

TEST_F(LogSignerTest, VerifyBadSTHSignature) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  // Too short.
  sth.CopyFrom(default_sth);
  EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

  string bad_signature = default_sth.signature().signature();
  bad_signature.erase(bad_signature.end() - 1);
  sth.mutable_signature()->set_signature(bad_signature);
  EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
            verifier_->VerifySTHSignature(sth));

  // Too long.
  // OpenSSL ECDSA Verify parses *up to* a given number of bytes,
  // rather than exactly the given number of bytes, and hence appending
  // garbage in the end still results in a valid signature.
  // sth.CopyFrom(default_sth);
  // EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

  // bad_signature = default_sth.signature().signature();
  // bad_signature.push_back(0x42);

  // sth.mutable_signature()->set_signature(bad_signature);
  // EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
  // verifier_->VerifySTHSignature(sth));

  // Flip the lsb of each byte one by one.
  for (size_t i = 0; i < default_sth.signature().signature().size(); ++i) {
    sth.CopyFrom(default_sth);
    EXPECT_EQ(LogSigVerifier::OK, verifier_->VerifySTHSignature(sth));

    bad_signature = default_sth.signature().signature();
    bad_signature[i] ^= 0x01;
    sth.mutable_signature()->set_signature(bad_signature);
    EXPECT_EQ(LogSigVerifier::INVALID_SIGNATURE,
              verifier_->VerifySTHSignature(sth));
  }
}

TEST_F(LogSignerTest, VerifyBadSerializedSCTSignature) {
  LogEntry default_entry;
  TestSigner::SetDefaults(&default_entry);
  SignedCertificateTimestamp default_sct, sct;
  TestSigner::SetDefaults(&default_sct);
  string serialized_sig = SerializedSignature(default_sct.signature());

  CHECK_EQ(ct::X509_ENTRY, default_entry.type());
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(), serialized_sig));
  // Too short.
  string bad_signature = serialized_sig.substr(0, serialized_sig.size() - 1);
  EXPECT_EQ(LogSigVerifier::SIGNATURE_TOO_SHORT,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(), bad_signature));
  // Too long.
  bad_signature = serialized_sig;
  bad_signature.push_back(0x42);
  EXPECT_EQ(LogSigVerifier::SIGNATURE_TOO_LONG,
            verifier_->VerifyV1CertSCTSignature(
                default_sct.timestamp(),
                default_entry.x509_entry().leaf_certificate(),
                default_sct.extensions(), bad_signature));

  // Flip the lsb of each byte one by one.
  for (size_t i = 0; i < serialized_sig.size(); ++i) {
    bad_signature = serialized_sig;
    bad_signature[i] ^= 0x01;
    // Error codes vary, depending on which byte was flipped.
    EXPECT_NE(LogSigVerifier::OK,
              verifier_->VerifyV1CertSCTSignature(
                  default_sct.timestamp(),
                  default_entry.x509_entry().leaf_certificate(),
                  default_sct.extensions(), bad_signature));
  }
}

TEST_F(LogSignerTest, VerifyBadSerializedSTHSignature) {
  SignedTreeHead default_sth, sth;
  TestSigner::SetDefaults(&default_sth);
  string serialized_sig = SerializedSignature(default_sth.signature());
  EXPECT_EQ(LogSigVerifier::OK,
            verifier_->VerifyV1STHSignature(default_sth.timestamp(),
                                            default_sth.tree_size(),
                                            default_sth.sha256_root_hash(),
                                            serialized_sig));
  // Too short.
  string bad_signature = serialized_sig.substr(0, serialized_sig.size() - 1);
  EXPECT_EQ(LogSigVerifier::SIGNATURE_TOO_SHORT,
            verifier_->VerifyV1STHSignature(default_sth.timestamp(),
                                            default_sth.tree_size(),
                                            default_sth.sha256_root_hash(),
                                            bad_signature));
  // Too long.
  bad_signature = serialized_sig;
  bad_signature.push_back(0x42);
  EXPECT_EQ(LogSigVerifier::SIGNATURE_TOO_LONG,
            verifier_->VerifyV1STHSignature(default_sth.timestamp(),
                                            default_sth.tree_size(),
                                            default_sth.sha256_root_hash(),
                                            bad_signature));

  // Flip the lsb of each byte one by one.
  for (size_t i = 0; i < serialized_sig.size(); ++i) {
    bad_signature = serialized_sig;
    bad_signature[i] ^= 0x01;
    // Error codes vary, depending on which byte was flipped.
    EXPECT_NE(LogSigVerifier::OK,
              verifier_->VerifyV1STHSignature(default_sth.timestamp(),
                                              default_sth.tree_size(),
                                              default_sth.sha256_root_hash(),
                                              bad_signature));
  }
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
