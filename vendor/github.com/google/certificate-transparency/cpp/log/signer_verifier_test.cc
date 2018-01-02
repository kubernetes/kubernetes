/* -*- indent-tabs-mode: nil -*- */
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <string>

#include "log/signer.h"
#include "log/test_signer.h"
#include "log/verifier.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/testing.h"
#include "util/util.h"

using cert_trans::Verifier;
using ct::DigitallySigned;
using std::string;

namespace cert_trans {
namespace {

const char kTestString[] = "abc";

class SignerVerifierTest : public ::testing::Test {
 protected:
  SignerVerifierTest() : signer_(NULL), verifier_(NULL) {
  }

  void SetUp() {
    signer_ = TestSigner::DefaultSigner();
    verifier_ = TestSigner::DefaultVerifier();
  }

  ~SignerVerifierTest() {
    delete signer_;
    delete verifier_;
  }

  static string SerializedSignature(const DigitallySigned& signature) {
    string serialized_sig;
    CHECK_EQ(SerializeResult::OK,
             Serializer::SerializeDigitallySigned(signature, &serialized_sig));
    return serialized_sig;
  }

  Signer* signer_;
  Verifier* verifier_;
};

// Check that the keys of the signer and verifier are consistent.
TEST_F(SignerVerifierTest, KeyID) {
  EXPECT_FALSE(signer_->KeyID().empty());
  EXPECT_EQ(signer_->KeyID(), verifier_->KeyID());
}

// TODO(ekasper): Check the example ECDSA signature for P-256, see
// http://www.nsa.gov/ia/_files/ecdsa.pdf, section D.1.

// Check that a well-known signature on well-know data verifies.
TEST_F(SignerVerifierTest, CheckSignature) {
  DigitallySigned signature;
  string data;
  signature.set_hash_algorithm(DigitallySigned::SHA256);
  signature.set_sig_algorithm(DigitallySigned::ECDSA);
  TestSigner::SetDefaults(&data, signature.mutable_signature());

  EXPECT_EQ(Verifier::OK, verifier_->Verify(data, signature));
}

// Check that a signature on a test string verifies.  Also check that
// successive signatures of the same data are different.
TEST_F(SignerVerifierTest, SignAndVerify) {
  // Sign the test string.
  DigitallySigned signature1;
  signer_->Sign(kTestString, &signature1);

  // Check that the signature verifies.
  EXPECT_EQ(DigitallySigned::SHA256, signature1.hash_algorithm());
  EXPECT_EQ(DigitallySigned::ECDSA, signature1.sig_algorithm());
  EXPECT_EQ(Verifier::OK, verifier_->Verify(kTestString, signature1));

  // Sign the test string a second time.
  DigitallySigned signature2;
  signer_->Sign(kTestString, &signature2);

  // Check that the signature is different but still verifies.
  EXPECT_NE(signature1.signature(), signature2.signature());
  EXPECT_EQ(DigitallySigned::SHA256, signature2.hash_algorithm());
  EXPECT_EQ(DigitallySigned::ECDSA, signature2.sig_algorithm());
  EXPECT_EQ(Verifier::OK, verifier_->Verify(kTestString, signature2));
}

// Check various error cases.
TEST_F(SignerVerifierTest, Errors) {
  DigitallySigned signature;
  string data;
  TestSigner::SetDefaults(&data, signature.mutable_signature());
  signature.set_hash_algorithm(DigitallySigned::SHA256);
  signature.set_sig_algorithm(DigitallySigned::ECDSA);
  EXPECT_EQ(Verifier::OK, verifier_->Verify(data, signature));

  const string good_signature = signature.signature();

  // Empty signature.
  signature.clear_signature();
  EXPECT_EQ(Verifier::INVALID_SIGNATURE, verifier_->Verify(data, signature));

  // Signature too short.
  signature.set_signature(
      good_signature.substr(0, good_signature.length() - 1));
  EXPECT_EQ(Verifier::INVALID_SIGNATURE, verifier_->Verify(data, signature));

  // Signature too long.
  // OpenSSL ECDSA Verify parses *up to* a given number of bytes,
  // rather than exactly the given number of bytes, and hence appending
  // garbage in the end still results in a valid signature.
  signature.set_signature(good_signature);
  signature.mutable_signature()->append(1, 'c');
  // EXPECT_EQ(Verifier::INVALID_SIGNATURE, verifier_->Verify(data,
  // signature));

  // Flip the lsb of each byte one by one.
  signature.set_signature(good_signature);
  for (size_t i = 0; i < good_signature.size(); ++i) {
    signature.mutable_signature()->at(i) ^= 0x01;
  }
  EXPECT_EQ(Verifier::INVALID_SIGNATURE, verifier_->Verify(data, signature));

  // Check algorithm mismatch.
  signature.set_signature(good_signature);
  signature.set_hash_algorithm(DigitallySigned::MD5);
  signature.set_sig_algorithm(DigitallySigned::ECDSA);
  EXPECT_EQ(Verifier::HASH_ALGORITHM_MISMATCH,
            verifier_->Verify(data, signature));

  signature.set_hash_algorithm(DigitallySigned::SHA256);
  signature.set_sig_algorithm(DigitallySigned::RSA);
  EXPECT_EQ(Verifier::SIGNATURE_ALGORITHM_MISMATCH,
            verifier_->Verify(data, signature));

  // Change data.
  data.append("foo");
  signature.set_sig_algorithm(DigitallySigned::ECDSA);
  EXPECT_EQ(Verifier::INVALID_SIGNATURE, verifier_->Verify(data, signature));
}

}  // namespace
}  // namespace cert_trans

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
