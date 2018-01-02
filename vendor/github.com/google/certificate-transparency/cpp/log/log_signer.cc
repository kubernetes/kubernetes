/* -*- indent-tabs-mode: nil -*- */
#include "log/log_signer.h"

#include <glog/logging.h>
#include <openssl/evp.h>
#include <openssl/opensslv.h>
#include <stdint.h>

#include "merkletree/serial_hasher.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/util.h"

using cert_trans::Verifier;
using ct::DigitallySigned;
using ct::LogEntry;
using ct::LogEntryType;
using ct::SignedCertificateTimestamp;
using ct::SignedTreeHead;
using std::string;

#if OPENSSL_VERSION_NUMBER < 0x10000000
#error "Need OpenSSL >= 1.0.0"
#endif

namespace {

LogSigVerifier::VerifyResult ConvertStatus(const Verifier::Status status) {
  switch (status) {
    case Verifier::OK:
      return LogSigVerifier::OK;
    case Verifier::HASH_ALGORITHM_MISMATCH:
      return LogSigVerifier::HASH_ALGORITHM_MISMATCH;
    case Verifier::SIGNATURE_ALGORITHM_MISMATCH:
      return LogSigVerifier::SIGNATURE_ALGORITHM_MISMATCH;
    case Verifier::INVALID_SIGNATURE:
      return LogSigVerifier::INVALID_SIGNATURE;
  }
  LOG(FATAL) << "Unexpected status " << status;
}

}  // namespace

LogSigner::LogSigner(EVP_PKEY* pkey) : cert_trans::Signer(pkey) {
}

LogSigner::~LogSigner() {
}

LogSigner::SignResult LogSigner::SignV1CertificateTimestamp(
    uint64_t timestamp, const string& leaf_certificate,
    const string& extensions, string* result) const {
  SignedCertificateTimestamp sct;
  sct.set_version(ct::V1);
  sct.set_timestamp(timestamp);
  sct.set_extensions(extensions);

  LogEntry entry;
  entry.set_type(ct::X509_ENTRY);
  entry.mutable_x509_entry()->set_leaf_certificate(leaf_certificate);

  string serialized_input;
  SerializeResult res =
      Serializer::SerializeSCTSignatureInput(sct, entry, &serialized_input);

  if (res != SerializeResult::OK)
    return GetSerializeError(res);

  DigitallySigned signature;
  Sign(serialized_input, &signature);
  CHECK_EQ(SerializeResult::OK,
           Serializer::SerializeDigitallySigned(signature, result));
  return OK;
}

LogSigner::SignResult LogSigner::SignV1PrecertificateTimestamp(
    uint64_t timestamp, const string& issuer_key_hash,
    const string& tbs_certificate, const string& extensions,
    string* result) const {
  SignedCertificateTimestamp sct;
  sct.set_version(ct::V1);
  sct.set_timestamp(timestamp);
  sct.set_extensions(extensions);

  LogEntry entry;
  entry.set_type(ct::PRECERT_ENTRY);
  entry.mutable_precert_entry()->mutable_pre_cert()->set_issuer_key_hash(
      issuer_key_hash);
  entry.mutable_precert_entry()->mutable_pre_cert()->set_tbs_certificate(
      tbs_certificate);

  string serialized_input;
  SerializeResult res =
      Serializer::SerializeSCTSignatureInput(sct, entry, &serialized_input);

  if (res != SerializeResult::OK)
    return GetSerializeError(res);

  DigitallySigned signature;
  Sign(serialized_input, &signature);
  CHECK_EQ(SerializeResult::OK,
           Serializer::SerializeDigitallySigned(signature, result));
  return OK;
}

LogSigner::SignResult LogSigner::SignCertificateTimestamp(
    const LogEntry& entry, SignedCertificateTimestamp* sct) const {
  CHECK(sct->has_timestamp())
      << "Attempt to sign an SCT with a missing timestamp";

  string serialized_input;
  SerializeResult res =
      Serializer::SerializeSCTSignatureInput(*sct, entry, &serialized_input);

  if (res != SerializeResult::OK)
    return GetSerializeError(res);
  Sign(serialized_input, sct->mutable_signature());
  sct->mutable_id()->set_key_id(KeyID());
  return OK;
}

LogSigner::SignResult LogSigner::SignV1TreeHead(uint64_t timestamp,
                                                int64_t tree_size,
                                                const string& root_hash,
                                                string* result) const {
  CHECK_GE(tree_size, 0);
  string serialized_sth;
  SerializeResult res =
      Serializer::SerializeV1STHSignatureInput(timestamp, tree_size, root_hash,
                                               &serialized_sth);

  if (res != SerializeResult::OK)
    return GetSerializeError(res);

  DigitallySigned signature;
  Sign(serialized_sth, &signature);
  CHECK_EQ(SerializeResult::OK,
           Serializer::SerializeDigitallySigned(signature, result));
  return OK;
}

LogSigner::SignResult LogSigner::SignTreeHead(SignedTreeHead* sth) const {
  string serialized_sth;
  SerializeResult res =
      Serializer::SerializeSTHSignatureInput(*sth, &serialized_sth);
  if (res != SerializeResult::OK)
    return GetSerializeError(res);
  Sign(serialized_sth, sth->mutable_signature());
  sth->mutable_id()->set_key_id(KeyID());
  return OK;
}

// static
LogSigner::SignResult LogSigner::GetSerializeError(SerializeResult result) {
  SignResult sign_result;
  switch (result) {
    case SerializeResult::INVALID_ENTRY_TYPE:
      sign_result = INVALID_ENTRY_TYPE;
      break;
    case SerializeResult::EMPTY_CERTIFICATE:
      sign_result = EMPTY_CERTIFICATE;
      break;
    case SerializeResult::CERTIFICATE_TOO_LONG:
      sign_result = CERTIFICATE_TOO_LONG;
      break;
    case SerializeResult::INVALID_HASH_LENGTH:
      sign_result = INVALID_HASH_LENGTH;
      break;
    case SerializeResult::UNSUPPORTED_VERSION:
      sign_result = UNSUPPORTED_VERSION;
      break;
    case SerializeResult::EXTENSIONS_TOO_LONG:
      sign_result = EXTENSIONS_TOO_LONG;
      break;
    default:
      LOG(FATAL) << "Unexpected Serializer error code " << result;
  }
  return sign_result;
}

LogSigVerifier::LogSigVerifier(EVP_PKEY* pkey) : Verifier(pkey) {
}

LogSigVerifier::~LogSigVerifier() {
}

LogSigVerifier::VerifyResult LogSigVerifier::VerifyV1CertSCTSignature(
    uint64_t timestamp, const string& leaf_cert, const string& extensions,
    const string& serialized_sig) const {
  DigitallySigned signature;
  DeserializeResult result =
      Deserializer::DeserializeDigitallySigned(serialized_sig, &signature);
  if (result != DeserializeResult::OK) {
    LOG(WARNING) << "DeserializeDigitallySigned returned " << result;
    return GetDeserializeSignatureError(result);
  }

  SignedCertificateTimestamp sct;
  sct.set_version(ct::V1);
  sct.set_timestamp(timestamp);
  sct.set_extensions(extensions);

  LogEntry entry;
  entry.set_type(ct::X509_ENTRY);
  entry.mutable_x509_entry()->set_leaf_certificate(leaf_cert);

  string serialized_sct;
  SerializeResult serialize_result =
      Serializer::SerializeSCTSignatureInput(sct, entry, &serialized_sct);

  if (serialize_result != SerializeResult::OK)
    return GetSerializeError(serialize_result);
  return ConvertStatus(Verify(serialized_sct, signature));
}

LogSigVerifier::VerifyResult LogSigVerifier::VerifyV1PrecertSCTSignature(
    uint64_t timestamp, const string& issuer_key_hash, const string& tbs_cert,
    const string& extensions, const string& serialized_sig) const {
  DigitallySigned signature;
  DeserializeResult result =
      Deserializer::DeserializeDigitallySigned(serialized_sig, &signature);
  if (result != DeserializeResult::OK)
    return GetDeserializeSignatureError(result);

  SignedCertificateTimestamp sct;
  sct.set_version(ct::V1);
  sct.set_timestamp(timestamp);
  sct.set_extensions(extensions);

  LogEntry entry;
  entry.set_type(ct::PRECERT_ENTRY);
  entry.mutable_precert_entry()->mutable_pre_cert()->set_issuer_key_hash(
      issuer_key_hash);
  entry.mutable_precert_entry()->mutable_pre_cert()->set_tbs_certificate(
      tbs_cert);

  string serialized_sct;
  SerializeResult serialize_result =
      Serializer::SerializeSCTSignatureInput(sct, entry, &serialized_sct);
  if (serialize_result != SerializeResult::OK)
    return GetSerializeError(serialize_result);
  return ConvertStatus(Verify(serialized_sct, signature));
}


LogSigVerifier::VerifyResult LogSigVerifier::VerifySCTSignature(
    const LogEntry& entry, const SignedCertificateTimestamp& sct) const {
  // Try to catch key mismatches early.
  if (sct.id().has_key_id() && sct.id().key_id() != KeyID()) {
    LOG(WARNING) << "Key ID mismatch, got: "
                 << util::HexString(sct.id().key_id())
                 << " expected: " << util::HexString(KeyID());
    return KEY_ID_MISMATCH;
  }

  string serialized_input;
  SerializeResult serialize_result =
      Serializer::SerializeSCTSignatureInput(sct, entry, &serialized_input);
  if (serialize_result != SerializeResult::OK)
    return GetSerializeError(serialize_result);
  return ConvertStatus(Verify(serialized_input, sct.signature()));
}

LogSigVerifier::VerifyResult LogSigVerifier::VerifyV1STHSignature(
    uint64_t timestamp, int64_t tree_size, const string& root_hash,
    const string& serialized_sig) const {
  CHECK_GE(tree_size, 0);
  DigitallySigned signature;
  DeserializeResult result =
      Deserializer::DeserializeDigitallySigned(serialized_sig, &signature);
  if (result != DeserializeResult::OK)
    return GetDeserializeSignatureError(result);

  string serialized_sth;
  SerializeResult serialize_result =
      Serializer::SerializeV1STHSignatureInput(timestamp, tree_size, root_hash,
                                               &serialized_sth);
  if (serialize_result != SerializeResult::OK)
    return GetSerializeError(serialize_result);
  return ConvertStatus(Verify(serialized_sth, signature));
}

LogSigVerifier::VerifyResult LogSigVerifier::VerifySTHSignature(
    const SignedTreeHead& sth) const {
  if (sth.id().has_key_id() && sth.id().key_id() != KeyID())
    return KEY_ID_MISMATCH;
  string serialized_sth;
  SerializeResult serialize_result =
      Serializer::SerializeSTHSignatureInput(sth, &serialized_sth);
  if (serialize_result != SerializeResult::OK)
    return GetSerializeError(serialize_result);
  return ConvertStatus(Verify(serialized_sth, sth.signature()));
}

// static
LogSigVerifier::VerifyResult LogSigVerifier::GetSerializeError(
    SerializeResult result) {
  VerifyResult verify_result;
  switch (result) {
    case SerializeResult::INVALID_ENTRY_TYPE:
      verify_result = INVALID_ENTRY_TYPE;
      break;
    case SerializeResult::EMPTY_CERTIFICATE:
      verify_result = EMPTY_CERTIFICATE;
      break;
    case SerializeResult::CERTIFICATE_TOO_LONG:
      verify_result = CERTIFICATE_TOO_LONG;
      break;
    case SerializeResult::INVALID_HASH_LENGTH:
      verify_result = INVALID_HASH_LENGTH;
      break;
    case SerializeResult::UNSUPPORTED_VERSION:
      verify_result = UNSUPPORTED_VERSION;
      break;
    case SerializeResult::EXTENSIONS_TOO_LONG:
      verify_result = EXTENSIONS_TOO_LONG;
      break;
    default:
      LOG(FATAL) << "Unexpected Serializer error code " << result;
  }
  return verify_result;
}

// static
LogSigVerifier::VerifyResult LogSigVerifier::GetDeserializeSignatureError(
    DeserializeResult result) {
  VerifyResult verify_result;
  switch (result) {
    case DeserializeResult::INPUT_TOO_SHORT:
      verify_result = SIGNATURE_TOO_SHORT;
      break;
    case DeserializeResult::INVALID_HASH_ALGORITHM:
      verify_result = INVALID_HASH_ALGORITHM;
      break;
    case DeserializeResult::INVALID_SIGNATURE_ALGORITHM:
      verify_result = INVALID_SIGNATURE_ALGORITHM;
      break;
    case DeserializeResult::INPUT_TOO_LONG:
      verify_result = SIGNATURE_TOO_LONG;
      break;
    default:
      LOG(FATAL) << "Unexpected Deserializer error code " << result;
  }
  return verify_result;
}
