/* -*- indent-tabs-mode: nil -*- */
#include "proto/cert_serializer.h"

#include <glog/logging.h>
#include <math.h>
#include <string>

#include "proto/ct.pb.h"
#include "proto/serializer.h"

using ct::DigitallySigned;
using ct::DigitallySigned_HashAlgorithm_IsValid;
using ct::DigitallySigned_SignatureAlgorithm_IsValid;
using ct::LogEntry;
using ct::LogEntryType_IsValid;
using ct::MerkleTreeLeaf;
using ct::PrecertChainEntry;
using ct::SignedCertificateTimestamp;
using ct::SignedCertificateTimestampList;
using ct::SthExtension;
using ct::SctExtension;
using ct::Version_IsValid;
using ct::X509ChainEntry;
using google::protobuf::RepeatedPtrField;
using std::string;


const size_t kMaxCertificateLength = (1 << 24) - 1;
const size_t kMaxCertificateChainLength = (1 << 24) - 1;


SerializeResult CheckCertificateFormat(const string& cert) {
  if (cert.empty()) {
    return SerializeResult::EMPTY_CERTIFICATE;
  }
  if (cert.size() > kMaxCertificateLength) {
    return SerializeResult::CERTIFICATE_TOO_LONG;
  }
  return SerializeResult::OK;
}


string CertV1LeafData(const LogEntry& entry) {
  switch (entry.type()) {
    // TODO(mhs): Because there is no X509_ENTRY_V2 we have to assume that
    // whichever of the cert fields is set defines the entry type. In other
    // words this is V2 if it has a CertInfo. Might be possible
    // to pass the type when the code that calls this is updated for V2.
    case ct::X509_ENTRY:
      CHECK(!entry.x509_entry().has_cert_info())
          << "Attempting to use a V1 serializer with a V2 LogEntry.";
      CHECK(entry.x509_entry().has_leaf_certificate())
          << "Missing leaf certificate";
      return entry.x509_entry().leaf_certificate();
    case ct::PRECERT_ENTRY:
      CHECK(!entry.x509_entry().has_cert_info())
          << "Attempting to use a V1 serializer with a V2 LogEntry.";
      CHECK(entry.precert_entry().pre_cert().has_tbs_certificate())
          << "Missing tbs certificate.";
      return entry.precert_entry().pre_cert().tbs_certificate();
    default:
      break;
  }
  LOG(FATAL) << "Invalid entry type " << entry.type();
}


SerializeResult SerializeV1CertSCTSignatureInput(uint64_t timestamp,
                                                 const string& certificate,
                                                 const string& extensions,
                                                 string* result) {
  SerializeResult res = CheckCertificateFormat(certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckExtensionsFormat(extensions);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V1, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::CERTIFICATE_TIMESTAMP,
                       Serializer::kSignatureTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::X509_ENTRY, Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteVarBytes(certificate, kMaxCertificateLength);
  serializer.WriteVarBytes(extensions, Serializer::kMaxExtensionsLength);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV1PrecertSCTSignatureInput(
    uint64_t timestamp, const string& issuer_key_hash,
    const string& tbs_certificate, const string& extensions, string* result) {
  SerializeResult res = CheckCertificateFormat(tbs_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckKeyHashFormat(issuer_key_hash);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckExtensionsFormat(extensions);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V1, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::CERTIFICATE_TIMESTAMP,
                       Serializer::kSignatureTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::PRECERT_ENTRY,
                       Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteFixedBytes(issuer_key_hash);
  serializer.WriteVarBytes(tbs_certificate, kMaxCertificateLength);
  serializer.WriteVarBytes(extensions, Serializer::kMaxExtensionsLength);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV1SCTSignatureInput(
    const SignedCertificateTimestamp& sct, const LogEntry& entry,
    string* result) {
  if (sct.version() != ct::V1) {
    return SerializeResult::UNSUPPORTED_VERSION;
  }
  switch (entry.type()) {
    case ct::X509_ENTRY:
      return SerializeV1CertSCTSignatureInput(
          sct.timestamp(), entry.x509_entry().leaf_certificate(),
          sct.extensions(), result);
    case ct::PRECERT_ENTRY:
      return SerializeV1PrecertSCTSignatureInput(
          sct.timestamp(), entry.precert_entry().pre_cert().issuer_key_hash(),
          entry.precert_entry().pre_cert().tbs_certificate(), sct.extensions(),
          result);
    default:
      return SerializeResult::INVALID_ENTRY_TYPE;
  }
}


SerializeResult SerializeV1CertSCTMerkleTreeLeaf(uint64_t timestamp,
                                                 const string& certificate,
                                                 const string& extensions,
                                                 string* result) {
  SerializeResult res = CheckCertificateFormat(certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckExtensionsFormat(extensions);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V1, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::TIMESTAMPED_ENTRY,
                       Serializer::kMerkleLeafTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::X509_ENTRY, Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteVarBytes(certificate, kMaxCertificateLength);
  serializer.WriteVarBytes(extensions, Serializer::kMaxExtensionsLength);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV1PrecertSCTMerkleTreeLeaf(
    uint64_t timestamp, const string& issuer_key_hash,
    const string& tbs_certificate, const string& extensions, string* result) {
  SerializeResult res = CheckCertificateFormat(tbs_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckKeyHashFormat(issuer_key_hash);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckExtensionsFormat(extensions);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V1, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::TIMESTAMPED_ENTRY,
                       Serializer::kMerkleLeafTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::PRECERT_ENTRY,
                       Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteFixedBytes(issuer_key_hash);
  serializer.WriteVarBytes(tbs_certificate, kMaxCertificateLength);
  serializer.WriteVarBytes(extensions, Serializer::kMaxExtensionsLength);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV1SCTMerkleTreeLeaf(
    const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
    string* result) {
  if (sct.version() != ct::V1) {
    return SerializeResult::UNSUPPORTED_VERSION;
  }
  switch (entry.type()) {
    case ct::X509_ENTRY:
      return SerializeV1CertSCTMerkleTreeLeaf(
          sct.timestamp(), entry.x509_entry().leaf_certificate(),
          sct.extensions(), result);
    case ct::PRECERT_ENTRY:
      return SerializeV1PrecertSCTMerkleTreeLeaf(
          sct.timestamp(), entry.precert_entry().pre_cert().issuer_key_hash(),
          entry.precert_entry().pre_cert().tbs_certificate(), sct.extensions(),
          result);
    default:
      break;
  }
  return SerializeResult::INVALID_ENTRY_TYPE;
}


DeserializeResult DeserializeV1SCTMerkleTreeLeaf(TLSDeserializer* des,
                                                 MerkleTreeLeaf* leaf) {
  CHECK_NOTNULL(des);
  CHECK_NOTNULL(leaf);

  unsigned int version;
  if (!des->ReadUint(Serializer::kVersionLengthInBytes, &version)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }

  if (version != ct::V1) {
    return DeserializeResult::UNSUPPORTED_VERSION;
  }
  leaf->set_version(ct::V1);

  unsigned int type;
  if (!des->ReadUint(Serializer::kMerkleLeafTypeLengthInBytes, &type)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  if (type != ct::TIMESTAMPED_ENTRY) {
    return DeserializeResult::UNKNOWN_LEAF_TYPE;
  }
  leaf->set_type(ct::TIMESTAMPED_ENTRY);

  ct::TimestampedEntry* const entry = leaf->mutable_timestamped_entry();

  uint64_t timestamp;
  if (!des->ReadUint(Serializer::kTimestampLengthInBytes, &timestamp)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  entry->set_timestamp(timestamp);

  unsigned int entry_type;
  if (!des->ReadUint(Serializer::kLogEntryTypeLengthInBytes, &entry_type)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }

  CHECK(LogEntryType_IsValid(entry_type));
  entry->set_entry_type(static_cast<ct::LogEntryType>(entry_type));

  switch (entry_type) {
    case ct::X509_ENTRY: {
      string x509;
      if (!des->ReadVarBytes(kMaxCertificateLength, &x509)) {
        return DeserializeResult::INPUT_TOO_SHORT;
      }
      entry->mutable_signed_entry()->set_x509(x509);
      return des->ReadExtensions(entry);
    }

    case ct::PRECERT_ENTRY: {
      string issuer_key_hash;
      if (!des->ReadFixedBytes(32, &issuer_key_hash)) {
        return DeserializeResult::INPUT_TOO_SHORT;
      }
      entry->mutable_signed_entry()->mutable_precert()->set_issuer_key_hash(
          issuer_key_hash);
      string tbs_certificate;
      if (!des->ReadVarBytes(kMaxCertificateLength, &tbs_certificate)) {
        return DeserializeResult::INPUT_TOO_SHORT;
      }
      entry->mutable_signed_entry()->mutable_precert()->set_tbs_certificate(
          tbs_certificate);
      return des->ReadExtensions(entry);
    }
  }

  LOG(FATAL) << "entry_type: " << entry_type;
  return DeserializeResult::UNKNOWN_LOGENTRY_TYPE;
}


// ----------------- V2 cert stuff ------------------------

string CertV2LeafData(const LogEntry& entry) {
  switch (entry.type()) {
    // TODO(mhs): Because there is no X509_ENTRY_V2 we have to assume that
    // whichever of the cert fields is set defines the entry type. In other
    // words this is V2 if it has a CertInfo. Might be possible
    // to pass the type when the code that calls this is updated for V2.
    case ct::X509_ENTRY:
      CHECK(!entry.x509_entry().has_leaf_certificate());
      CHECK(entry.x509_entry().has_cert_info());
      CHECK(entry.x509_entry().cert_info().has_tbs_certificate())
          << "Missing V2 leaf certificate";
      return entry.x509_entry().cert_info().tbs_certificate();
    case ct::PRECERT_ENTRY_V2:
      // Must not have both v1 and v2 entries set
      CHECK(!entry.precert_entry().has_pre_cert());
      CHECK(entry.precert_entry().cert_info().has_tbs_certificate())
          << "Missing tbs certificate (V2)";
      return entry.precert_entry().cert_info().tbs_certificate();
    default:
      break;
  }
  LOG(FATAL) << "Invalid entry type " << entry.type();
}


// static
SerializeResult SerializeV2CertSCTSignatureInput(
    uint64_t timestamp, const string& issuer_key_hash,
    const string& tbs_certificate,
    const RepeatedPtrField<ct::SctExtension>& sct_extension, string* result) {
  SerializeResult res = CheckCertificateFormat(tbs_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckSctExtensionsFormat(sct_extension);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V2, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::CERTIFICATE_TIMESTAMP,
                       Serializer::kSignatureTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::X509_ENTRY, Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteFixedBytes(issuer_key_hash);
  serializer.WriteVarBytes(tbs_certificate, kMaxCertificateLength);
  serializer.WriteSctExtension(sct_extension);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


// static
SerializeResult SerializeV2PrecertSCTSignatureInput(
    uint64_t timestamp, const string& issuer_key_hash,
    const string& tbs_certificate,
    const RepeatedPtrField<ct::SctExtension>& sct_extension, string* result) {
  SerializeResult res = CheckCertificateFormat(tbs_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckSctExtensionsFormat(sct_extension);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V2, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::CERTIFICATE_TIMESTAMP,
                       Serializer::kSignatureTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::PRECERT_ENTRY_V2,
                       Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteFixedBytes(issuer_key_hash);
  serializer.WriteVarBytes(tbs_certificate, kMaxCertificateLength);
  serializer.WriteSctExtension(sct_extension);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}

// static
SerializeResult SerializeV2SCTSignatureInput(
    const SignedCertificateTimestamp& sct, const LogEntry& entry,
    string* result) {
  if (sct.version() != ct::V2) {
    return SerializeResult::UNSUPPORTED_VERSION;
  }
  switch (entry.type()) {
    case ct::X509_ENTRY:
      return SerializeV2CertSCTSignatureInput(
          sct.timestamp(), entry.x509_entry().cert_info().issuer_key_hash(),
          entry.x509_entry().cert_info().tbs_certificate(),
          sct.sct_extension(), result);
    case ct::PRECERT_ENTRY_V2:
      return SerializeV2PrecertSCTSignatureInput(
          sct.timestamp(), entry.precert_entry().cert_info().issuer_key_hash(),
          entry.precert_entry().cert_info().tbs_certificate(),
          sct.sct_extension(), result);
    default:
      break;
  }
  return SerializeResult::INVALID_ENTRY_TYPE;
}


SerializeResult SerializeV2CertSCTMerkleTreeLeaf(
    uint64_t timestamp, const string& issuer_key_hash,
    const string& tbs_certificate,
    const RepeatedPtrField<SctExtension>& sct_extension, string* result) {
  SerializeResult res = CheckCertificateFormat(tbs_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckSctExtensionsFormat(sct_extension);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V2, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::TIMESTAMPED_ENTRY,
                       Serializer::kMerkleLeafTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::X509_ENTRY, Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteFixedBytes(issuer_key_hash);
  serializer.WriteVarBytes(tbs_certificate, kMaxCertificateLength);
  serializer.WriteSctExtension(sct_extension);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV2PrecertSCTMerkleTreeLeaf(
    uint64_t timestamp, const string& issuer_key_hash,
    const string& tbs_certificate,
    const google::protobuf::RepeatedPtrField<ct::SctExtension>& sct_extension,
    string* result) {
  SerializeResult res = CheckCertificateFormat(tbs_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckKeyHashFormat(issuer_key_hash);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckSctExtensionsFormat(sct_extension);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V2, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::TIMESTAMPED_ENTRY,
                       Serializer::kMerkleLeafTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(ct::PRECERT_ENTRY_V2,
                       Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteFixedBytes(issuer_key_hash);
  serializer.WriteVarBytes(tbs_certificate, kMaxCertificateLength);
  serializer.WriteSctExtension(sct_extension);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV2SCTMerkleTreeLeaf(
    const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
    string* result) {
  CHECK_EQ(ct::V2, sct.version());
  switch (entry.type()) {
    case ct::X509_ENTRY:
      return SerializeV2CertSCTMerkleTreeLeaf(
          sct.timestamp(), entry.x509_entry().cert_info().issuer_key_hash(),
          entry.x509_entry().cert_info().tbs_certificate(),
          sct.sct_extension(), result);
    case ct::PRECERT_ENTRY_V2:
      return SerializeV2PrecertSCTMerkleTreeLeaf(
          sct.timestamp(), entry.precert_entry().cert_info().issuer_key_hash(),
          entry.precert_entry().cert_info().tbs_certificate(),
          sct.sct_extension(), result);
    default:
      break;
  }
  return SerializeResult::INVALID_ENTRY_TYPE;
}


DeserializeResult DeserializeV2SCTMerkleTreeLeaf(TLSDeserializer* des,
                                                 MerkleTreeLeaf* leaf) {
  CHECK_NOTNULL(des);
  CHECK_NOTNULL(leaf);

  unsigned int version;
  if (!des->ReadUint(Serializer::kVersionLengthInBytes, &version)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }

  if (version != ct::V2) {
    return DeserializeResult::UNSUPPORTED_VERSION;
  }
  leaf->set_version(ct::V2);

  unsigned int type;
  if (!des->ReadUint(Serializer::kMerkleLeafTypeLengthInBytes, &type)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  if (type != ct::TIMESTAMPED_ENTRY) {
    return DeserializeResult::UNKNOWN_LEAF_TYPE;
  }
  leaf->set_type(ct::TIMESTAMPED_ENTRY);

  ct::TimestampedEntry* const entry = leaf->mutable_timestamped_entry();

  uint64_t timestamp;
  if (!des->ReadUint(Serializer::kTimestampLengthInBytes, &timestamp)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  entry->set_timestamp(timestamp);

  unsigned int entry_type;
  if (!des->ReadUint(Serializer::kLogEntryTypeLengthInBytes, &entry_type)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }

  CHECK(LogEntryType_IsValid(entry_type));
  entry->set_entry_type(static_cast<ct::LogEntryType>(entry_type));

  switch (entry_type) {
    // In V2 both X509 and Precert entries use CertInfo
    case ct::X509_ENTRY:
    case ct::PRECERT_ENTRY_V2: {
      string issuer_key_hash;
      if (!des->ReadFixedBytes(32, &issuer_key_hash)) {
        return DeserializeResult::INPUT_TOO_SHORT;
      }
      entry->mutable_signed_entry()->mutable_cert_info()->set_issuer_key_hash(
          issuer_key_hash);
      string tbs_certificate;
      if (!des->ReadVarBytes(kMaxCertificateLength, &tbs_certificate)) {
        return DeserializeResult::INPUT_TOO_SHORT;
      }
      entry->mutable_signed_entry()->mutable_cert_info()->set_tbs_certificate(
          tbs_certificate);
      return des->ReadExtensions(entry);
    }

    case ct::UNKNOWN_ENTRY_TYPE: {
      // handled below.
      break;
    }
  }
  LOG(FATAL) << "entry_type: " << entry_type;
  return DeserializeResult::UNKNOWN_LOGENTRY_TYPE;
}


void ConfigureSerializerForV1CT() {
  Serializer::ConfigureV1(CertV1LeafData, SerializeV1SCTSignatureInput,
                          SerializeV1SCTMerkleTreeLeaf);
  Deserializer::Configure(DeserializeV1SCTMerkleTreeLeaf);
}


void ConfigureSerializerForV2CT() {
  Serializer::ConfigureV2(CertV2LeafData, SerializeV2SCTSignatureInput,
                          SerializeV2SCTMerkleTreeLeaf);
  Deserializer::Configure(DeserializeV2SCTMerkleTreeLeaf);
}


SerializeResult SerializeX509Chain(const ct::X509ChainEntry& entry,
                                   std::string* result) {
  return SerializeX509ChainV1(entry.certificate_chain(), result);
}


SerializeResult SerializeX509ChainV1(const repeated_string& certificate_chain,
                                     std::string* result) {
  return Serializer::SerializeList(certificate_chain, kMaxCertificateLength,
                                   kMaxCertificateChainLength, result);
}


SerializeResult SerializePrecertChainEntry(const ct::PrecertChainEntry& entry,
                                           std::string* result) {
  return SerializePrecertChainEntry(entry.pre_certificate(),
                                    entry.precertificate_chain(), result);
}


SerializeResult SerializePrecertChainEntry(
    const std::string& pre_certificate,
    const repeated_string& precertificate_chain, std::string* result) {
  TLSSerializer serializer;
  if (pre_certificate.size() > kMaxCertificateLength) {
    return SerializeResult::CERTIFICATE_TOO_LONG;
  }
  if (pre_certificate.empty()) {
    return SerializeResult::EMPTY_CERTIFICATE;
  }

  serializer.WriteVarBytes(pre_certificate, kMaxCertificateLength);

  SerializeResult res =
      serializer.WriteList(precertificate_chain, kMaxCertificateLength,
                           kMaxCertificateChainLength);
  if (res != SerializeResult::OK) {
    return res;
  }
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV1SignedCertEntryWithType(
    const std::string& leaf_certificate, std::string* result) {
  SerializeResult res = CheckCertificateFormat(leaf_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::X509_ENTRY, Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteVarBytes(leaf_certificate, kMaxCertificateLength);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult SerializeV1SignedPrecertEntryWithType(
    const std::string& issuer_key_hash, const std::string& tbs_certificate,
    std::string* result) {
  SerializeResult res = CheckCertificateFormat(tbs_certificate);
  if (res != SerializeResult::OK) {
    return res;
  }
  res = CheckKeyHashFormat(issuer_key_hash);
  if (res != SerializeResult::OK) {
    return res;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::PRECERT_ENTRY,
                       Serializer::kLogEntryTypeLengthInBytes);
  serializer.WriteFixedBytes(issuer_key_hash);
  serializer.WriteVarBytes(tbs_certificate, kMaxCertificateLength);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


// static
DeserializeResult DeserializeX509Chain(const std::string& in,
                                       X509ChainEntry* x509_chain_entry) {
  // Empty list is ok.
  x509_chain_entry->clear_certificate_chain();
  return Deserializer::DeserializeList(
      in, kMaxCertificateChainLength, kMaxCertificateLength,
      x509_chain_entry->mutable_certificate_chain());
}


// static
DeserializeResult DeserializePrecertChainEntry(
    const std::string& in, ct::PrecertChainEntry* precert_chain_entry) {
  TLSDeserializer deserializer(in);
  if (!deserializer.ReadVarBytes(
          kMaxCertificateLength,
          precert_chain_entry->mutable_pre_certificate())) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  precert_chain_entry->clear_precertificate_chain();
  DeserializeResult res = deserializer.ReadList(
      kMaxCertificateChainLength, kMaxCertificateLength,
      precert_chain_entry->mutable_precertificate_chain());
  if (res != DeserializeResult::OK) {
    return res;
  }
  if (!deserializer.ReachedEnd()) {
    return DeserializeResult::INPUT_TOO_LONG;
  }
  return DeserializeResult::OK;
}
