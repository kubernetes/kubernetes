/* -*- indent-tabs-mode: nil -*- */
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <memory>
#include <string>

#include "log/cert.h"
#include "log/cms_verifier.h"
#include "log/ct_extensions.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/util.h"

using cert_trans::Cert;
using cert_trans::CmsVerifier;
using cert_trans::ScopedBIO;
using std::string;
using std::unique_ptr;
using std::vector;
using util::testing::StatusIs;

// Self-signed
static const char kCaCert[] = "ca-cert.pem";
// Issued by ca-cert.pem
static const char kLeafCert[] = "test-cert.pem";
// Issued by ca-cert.pem
static const char kIntermediateCert[] = "intermediate-cert.pem";

// A DER file containing a CMS signed message wrapping data that is not
// valid DER
static const char kCmsSignedDataTest2[] = "cms_test2.der";
// A DER file containing a CMS signed message wrapping a DER encoded
// certificate for test case 3 (valid signature, same signer as cert)
static const char kCmsSignedDataTest3[] = "cms_test3.der";
// A DER file with a CMS signed message but not signed by the same
// key as the certificate it contains in the payload
static const char kCmsSignedDataTest4[] = "cms_test4.der";
// A DER file with a CMS signed message with intermediate as signer and
// issuer of the embedded cert
static const char kCmsSignedDataTest5[] = "cms_test5.der";

// Subject name we expect in our embedded certificate CMS tests
static const char kCmsTestSubject[] =
    "CN=?.example.com, C=GB, ST=Wales, "
    "L=Erw Wen, O=Certificate Transparency";

namespace {

class CmsVerifierTest : public ::testing::Test {
 protected:
  CmsVerifierTest()
      : cert_dir_(FLAGS_test_srcdir + "/test/testdata"),
        cert_dir_v2_(FLAGS_test_srcdir + "/test/testdata/v2/") {
  }

  string ca_pem_;
  string intermediate_pem_;
  string leaf_pem_;
  const string cert_dir_;
  const string cert_dir_v2_;
  CmsVerifier verifier_;

  void SetUp() {
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kLeafCert, &leaf_pem_))
        << "Could not read test data from " << cert_dir_
        << ". Wrong --test_srcdir?";
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kIntermediateCert,
                             &intermediate_pem_));
    CHECK(util::ReadTextFile(cert_dir_ + "/" + kCaCert, &ca_pem_));
  }
};


BIO* OpenTestFileBio(const string& filename) {
  BIO* der_bio = BIO_new_file(filename.c_str(), "r");

  CHECK_NOTNULL(der_bio);

  return der_bio;
}


TEST_F(CmsVerifierTest, CmsSignTestCase2) {
  // In this test the embedded data is not a certificate in DER format
  // but it doesn't get unpacked and the signature is valid.
  Cert ca(ca_pem_);

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest2));
  ASSERT_NE(bio.get(), nullptr);
  EXPECT_TRUE(verifier_.IsCmsSignedByCert(bio.get(), ca).ValueOrDie());
}


TEST_F(CmsVerifierTest, CmsSignTestCase3) {
  // The CMS should be signed by the CA that signed the cert
  Cert ca(ca_pem_);

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest3));
  ASSERT_NE(bio.get(), nullptr);
  EXPECT_TRUE(verifier_.IsCmsSignedByCert(bio.get(), ca).ValueOrDie());
}


TEST_F(CmsVerifierTest, CmsSignTestCase4) {
  // The CMS is not signed by the CA that signed the cert it contains
  Cert ca(ca_pem_);

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest4));
  ASSERT_NE(bio.get(), nullptr);
  EXPECT_FALSE(verifier_.IsCmsSignedByCert(bio.get(), ca).ValueOrDie());
}


TEST_F(CmsVerifierTest, CmsVerifyTestCase2) {
  // For this test the embedded cert is invalid DER but CMS signed by the CA
  Cert cert(ca_pem_);
  ASSERT_TRUE(cert.IsLoaded());

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest2));
  unique_ptr<Cert> unpacked_cert(
      verifier_.UnpackCmsSignedCertificate(bio.get(), cert));

  ASSERT_FALSE(unpacked_cert->IsLoaded());
}


TEST_F(CmsVerifierTest, CmsVerifyTestCase3) {
  // For this test the embedded cert is signed by the CA
  Cert cert(ca_pem_);
  ASSERT_TRUE(cert.IsLoaded());

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest3));
  unique_ptr<Cert> unpacked_cert(
      verifier_.UnpackCmsSignedCertificate(bio.get(), cert));

  ASSERT_FALSE(unpacked_cert->HasBasicConstraintCATrue().ValueOrDie());
  ASSERT_TRUE(
      unpacked_cert->HasExtension(NID_authority_key_identifier).ValueOrDie());
  // We built the embedded cert with redaction so this helps to prove
  // that it was correctly unpacked
  ASSERT_OK(unpacked_cert->IsValidWildcardRedaction());
  ASSERT_EQ(kCmsTestSubject, unpacked_cert->PrintSubjectName());
}


TEST_F(CmsVerifierTest, CmsVerifyTestCase4) {
  // For this test the embedded cert is signed by the intermediate CA
  Cert cert(ca_pem_);
  ASSERT_TRUE(cert.IsLoaded());

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest4));
  unique_ptr<Cert> unpacked_cert(
      verifier_.UnpackCmsSignedCertificate(bio.get(), cert));

  ASSERT_FALSE(unpacked_cert->IsLoaded());
}


TEST_F(CmsVerifierTest, CmsVerifyTestCase5) {
  // For this test the embedded cert is signed by the intermediate
  Cert cert(intermediate_pem_);
  ASSERT_TRUE(cert.IsLoaded());

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest5));
  unique_ptr<Cert> unpacked_cert(
      verifier_.UnpackCmsSignedCertificate(bio.get(), cert));

  ASSERT_FALSE(unpacked_cert->HasBasicConstraintCATrue().ValueOrDie());
  ASSERT_TRUE(
      unpacked_cert->HasExtension(NID_authority_key_identifier).ValueOrDie());
  // We built the embedded cert with redaction so this helps to prove
  // that it was correctly unpacked
  ASSERT_OK(unpacked_cert->IsValidWildcardRedaction());
  ASSERT_EQ(kCmsTestSubject, unpacked_cert->PrintSubjectName());
}


TEST_F(CmsVerifierTest, CmsVerifyTestCase7) {
  // For this test the embedded cert is signed by the intermediate
  Cert cert(leaf_pem_);
  ASSERT_TRUE(cert.IsLoaded());

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest5));
  unique_ptr<Cert> unpacked_cert(
      verifier_.UnpackCmsSignedCertificate(bio.get(), cert));

  ASSERT_FALSE(unpacked_cert->IsLoaded());
}


TEST_F(CmsVerifierTest, CmsVerifyTestCase8) {
  // For this test the embedded cert is signed by the intermediate
  Cert cert(ca_pem_);
  ASSERT_TRUE(cert.IsLoaded());

  ScopedBIO bio(OpenTestFileBio(cert_dir_v2_ + kCmsSignedDataTest5));
  unique_ptr<Cert> unpacked_cert(
      verifier_.UnpackCmsSignedCertificate(bio.get(), cert));

  ASSERT_FALSE(unpacked_cert->IsLoaded());
}

}  // namespace


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  OpenSSL_add_all_algorithms();
  ERR_load_crypto_strings();
  cert_trans::LoadCtExtensions();
  return RUN_ALL_TESTS();
}
