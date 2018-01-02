/* -*- indent-tabs-mode: nil -*- */
#include "log/verifier.h"

#include <glog/logging.h>
#include <openssl/evp.h>
#include <openssl/opensslv.h>
#include <stdint.h>

#include "merkletree/serial_hasher.h"
#include "proto/ct.pb.h"
#include "util/util.h"

#if OPENSSL_VERSION_NUMBER < 0x10000000
#error "Need OpenSSL >= 1.0.0"
#endif

using ct::DigitallySigned;

namespace cert_trans {

Verifier::Verifier(EVP_PKEY* pkey) : pkey_(CHECK_NOTNULL(pkey)) {
  switch (pkey_->type) {
    case EVP_PKEY_EC:
      hash_algo_ = DigitallySigned::SHA256;
      sig_algo_ = DigitallySigned::ECDSA;
      break;
    case EVP_PKEY_RSA:
      hash_algo_ = DigitallySigned::SHA256;
      sig_algo_ = DigitallySigned::RSA;
      break;
    default:
      LOG(FATAL) << "Unsupported key type " << pkey_->type;
  }
  key_id_ = ComputeKeyID(pkey_.get());
}

std::string Verifier::KeyID() const {
  return key_id_;
}

Verifier::Status Verifier::Verify(const std::string& input,
                                  const DigitallySigned& signature) const {
  if (signature.hash_algorithm() != hash_algo_)
    return HASH_ALGORITHM_MISMATCH;
  if (signature.sig_algorithm() != sig_algo_)
    return SIGNATURE_ALGORITHM_MISMATCH;
  if (!RawVerify(input, signature.signature()))
    return INVALID_SIGNATURE;
  return OK;
}

// static
std::string Verifier::ComputeKeyID(EVP_PKEY* pkey) {
  // i2d_PUBKEY sets the algorithm and (for ECDSA) named curve parameter and
  // encodes the key as an X509_PUBKEY (i.e., subjectPublicKeyInfo).
  int buf_len = i2d_PUBKEY(pkey, NULL);
  CHECK_GT(buf_len, 0);
  unsigned char* buf = new unsigned char[buf_len];
  unsigned char* p = buf;
  CHECK_EQ(i2d_PUBKEY(pkey, &p), buf_len);
  const std::string keystring(reinterpret_cast<char*>(buf), buf_len);
  const std::string ret(Sha256Hasher::Sha256Digest(keystring));
  delete[] buf;
  return ret;
}

Verifier::Verifier()
    : hash_algo_(DigitallySigned::NONE),
      sig_algo_(DigitallySigned::ANONYMOUS) {
}

bool Verifier::RawVerify(const std::string& data,
                         const std::string& sig_string) const {
  EVP_MD_CTX ctx;
  EVP_MD_CTX_init(&ctx);
  // NOTE: this syntax for setting the hash function requires OpenSSL >= 1.0.0.
  CHECK_EQ(1, EVP_VerifyInit(&ctx, EVP_sha256()));
  CHECK_EQ(1, EVP_VerifyUpdate(&ctx, data.data(), data.size()));
  bool ret = (EVP_VerifyFinal(&ctx, reinterpret_cast<const unsigned char*>(
                                        sig_string.data()),
                              sig_string.size(), pkey_.get()) == 1);
  EVP_MD_CTX_cleanup(&ctx);
  return ret;
}

}  // namespace cert_trans
