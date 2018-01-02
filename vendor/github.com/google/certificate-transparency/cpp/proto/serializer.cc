/* -*- indent-tabs-mode: nil -*- */
#include "proto/serializer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <math.h>
#include <string>

#include "proto/ct.pb.h"

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
using std::function;
using std::string;

const size_t Serializer::kMaxSignatureLength = (1 << 16) - 1;
const size_t Serializer::kMaxV2ExtensionType = (1 << 16) - 1;
const size_t Serializer::kMaxV2ExtensionsCount = (1 << 16) - 2;
const size_t Serializer::kMaxExtensionsLength = (1 << 16) - 1;
const size_t Serializer::kMaxSerializedSCTLength = (1 << 16) - 1;
const size_t Serializer::kMaxSCTListLength = (1 << 16) - 1;

const size_t Serializer::kLogEntryTypeLengthInBytes = 2;
const size_t Serializer::kSignatureTypeLengthInBytes = 1;
const size_t Serializer::kHashAlgorithmLengthInBytes = 1;
const size_t Serializer::kSigAlgorithmLengthInBytes = 1;
const size_t Serializer::kVersionLengthInBytes = 1;
const size_t Serializer::kKeyIDLengthInBytes = 32;
const size_t Serializer::kMerkleLeafTypeLengthInBytes = 1;
const size_t Serializer::kKeyHashLengthInBytes = 32;
const size_t Serializer::kTimestampLengthInBytes = 8;

DEFINE_bool(allow_reconfigure_serializer_test_only, false,
            "Allow tests to reconfigure the serializer multiple times.");

// TODO(pphaneuf): This is just to avoid causing diff churn while
// refactoring. Functions for internal use only should be put together
// in an anonymous namespace.
SerializeResult CheckSignatureFormat(const DigitallySigned& sig);
SerializeResult CheckSthExtensionsFormat(
    const repeated_sth_extension& extension);


namespace {


function<string(const ct::LogEntry&)> leaf_data;

function<SerializeResult(const ct::SignedCertificateTimestamp& sct,
                         const ct::LogEntry& entry, std::string* result)>
    serialize_sct_sig_input;

function<SerializeResult(const ct::SignedCertificateTimestamp& sct,
                         const ct::LogEntry& entry, std::string* result)>
    serialize_sct_merkle_leaf;

function<SerializeResult(uint64_t timestamp, int64_t tree_size,
                         const std::string& root_hash, std::string* result)>
    serialize_sth_sig_input_v1;

function<SerializeResult(uint64_t timestamp, int64_t tree_size,
                         const std::string& root_hash,
                         const repeated_sth_extension& sth_extension,
                         const std::string& log_id, std::string* result)>
    serialize_sth_sig_input_v2;

function<DeserializeResult(TLSDeserializer* d, ct::MerkleTreeLeaf* leaf)>
    read_merkle_tree_leaf;


}  // namespace


std::ostream& operator<<(std::ostream& stream, const SerializeResult& r) {
  switch (r) {
    case SerializeResult::OK:
      return stream << "OK";
    case SerializeResult::INVALID_ENTRY_TYPE:
      return stream << "INVALID_ENTRY_TYPE";
    case SerializeResult::EMPTY_CERTIFICATE:
      return stream << "EMPTY_CERTIFICATE";
    case SerializeResult::CERTIFICATE_TOO_LONG:
      return stream << "CERTIFICATE_TOO_LONG";
    case SerializeResult::CERTIFICATE_CHAIN_TOO_LONG:
      return stream << "CERTIFICATE_CHAIN_TOO_LONG";
    case SerializeResult::INVALID_HASH_ALGORITHM:
      return stream << "INVALID_HASH_ALGORITHM";
    case SerializeResult::INVALID_SIGNATURE_ALGORITHM:
      return stream << "INVALID_SIGNATURE_ALGORITHM";
    case SerializeResult::SIGNATURE_TOO_LONG:
      return stream << "SIGNATURE_TOO_LONG";
    case SerializeResult::INVALID_HASH_LENGTH:
      return stream << "INVALID_HASH_LENGTH";
    case SerializeResult::EMPTY_PRECERTIFICATE_CHAIN:
      return stream << "EMPTY_PRECERTIFICATE_CHAIN";
    case SerializeResult::UNSUPPORTED_VERSION:
      return stream << "UNSUPPORTED_VERSION";
    case SerializeResult::EXTENSIONS_TOO_LONG:
      return stream << "EXTENSIONS_TOO_LONG";
    case SerializeResult::INVALID_KEYID_LENGTH:
      return stream << "INVALID_KEYID_LENGTH";
    case SerializeResult::EMPTY_LIST:
      return stream << "EMPTY_LIST";
    case SerializeResult::EMPTY_ELEM_IN_LIST:
      return stream << "EMPTY_ELEM_IN_LIST";
    case SerializeResult::LIST_ELEM_TOO_LONG:
      return stream << "LIST_ELEM_TOO_LONG";
    case SerializeResult::LIST_TOO_LONG:
      return stream << "LIST_TOO_LONG";
    case SerializeResult::EXTENSIONS_NOT_ORDERED:
      return stream << "EXTENSIONS_NOT_ORDERED";
  }
}


std::ostream& operator<<(std::ostream& stream, const DeserializeResult& r) {
  switch (r) {
    case DeserializeResult::OK:
      return stream << "OK";
    case DeserializeResult::INPUT_TOO_SHORT:
      return stream << "INPUT_TOO_SHORT";
    case DeserializeResult::INVALID_HASH_ALGORITHM:
      return stream << "INVALID_HASH_ALGORITHM";
    case DeserializeResult::INVALID_SIGNATURE_ALGORITHM:
      return stream << "INVALID_SIGNATURE_ALGORITHM";
    case DeserializeResult::INPUT_TOO_LONG:
      return stream << "INPUT_TOO_LONG";
    case DeserializeResult::UNSUPPORTED_VERSION:
      return stream << "UNSUPPORTED_VERSION";
    case DeserializeResult::INVALID_LIST_ENCODING:
      return stream << "INVALID_LIST_ENCODING";
    case DeserializeResult::EMPTY_LIST:
      return stream << "EMPTY_LIST";
    case DeserializeResult::EMPTY_ELEM_IN_LIST:
      return stream << "EMPTY_ELEM_IN_LIST";
    case DeserializeResult::UNKNOWN_LEAF_TYPE:
      return stream << "UNKNOWN_LEAF_TYPE";
    case DeserializeResult::UNKNOWN_LOGENTRY_TYPE:
      return stream << "UNKNOWN_LOGENTRY_TYPE";
    case DeserializeResult::EXTENSIONS_TOO_LONG:
      return stream << "EXTENSIONS_TOO_LONG";
    case DeserializeResult::EXTENSIONS_NOT_ORDERED:
      return stream << "EXTENSIONS_NOT_ORDERED";
  }
}


// Returns the number of bytes needed to store a value up to max_length.
size_t PrefixLength(size_t max_length) {
  CHECK_GT(max_length, 0U);
  return ceil(log2(max_length) / float(8));
}


// static
string Serializer::LeafData(const LogEntry& entry) {
  CHECK(leaf_data);
  return leaf_data(entry);
}


// static
SerializeResult Serializer::SerializeV1STHSignatureInput(
    uint64_t timestamp, int64_t tree_size, const string& root_hash,
    string* result) {
  CHECK(result);
  CHECK(serialize_sth_sig_input_v1);
  return serialize_sth_sig_input_v1(timestamp, tree_size, root_hash, result);
}


static SerializeResult SerializeV1STHSignatureInput(uint64_t timestamp,
                                                    int64_t tree_size,
                                                    const string& root_hash,
                                                    string* result) {
  CHECK_GE(tree_size, 0);
  if (root_hash.size() != 32)
    return SerializeResult::INVALID_HASH_LENGTH;
  TLSSerializer serializer;
  serializer.WriteUint(ct::V1, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::TREE_HEAD, Serializer::kSignatureTypeLengthInBytes);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(tree_size, 8);
  serializer.WriteFixedBytes(root_hash);
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


// static
SerializeResult Serializer::SerializeV2STHSignatureInput(
    uint64_t timestamp, int64_t tree_size, const string& root_hash,
    const repeated_sth_extension& sth_extension, const string& log_id,
    string* result) {
  CHECK(result);
  CHECK(serialize_sth_sig_input_v2);
  return serialize_sth_sig_input_v2(timestamp, tree_size, root_hash,
                                     sth_extension, log_id, result);
}


static SerializeResult SerializeV2STHSignatureInput(
    uint64_t timestamp, int64_t tree_size, const string& root_hash,
    const RepeatedPtrField<SthExtension>& sth_extension, const string& log_id,
    string* result) {
  CHECK_GE(tree_size, 0);
  if (root_hash.size() != 32) {
    return SerializeResult::INVALID_HASH_LENGTH;
  }
  SerializeResult res = CheckSthExtensionsFormat(sth_extension);
  if (res != SerializeResult::OK) {
    return res;
  }
  if (log_id.size() != Serializer::kKeyIDLengthInBytes) {
    return SerializeResult::INVALID_KEYID_LENGTH;
  }
  TLSSerializer serializer;
  serializer.WriteUint(ct::V2, Serializer::kVersionLengthInBytes);
  serializer.WriteUint(ct::TREE_HEAD, Serializer::kSignatureTypeLengthInBytes);
  serializer.WriteFixedBytes(log_id);
  serializer.WriteUint(timestamp, Serializer::kTimestampLengthInBytes);
  serializer.WriteUint(tree_size, 8);
  serializer.WriteFixedBytes(root_hash);
  // V2 STH can have multiple extensions
  serializer.WriteUint(sth_extension.size(), 2);
  for (auto it = sth_extension.begin(); it != sth_extension.end(); ++it) {
    serializer.WriteUint(it->sth_extension_type(), 2);
    serializer.WriteVarBytes(it->sth_extension_data(),
                             Serializer::kMaxExtensionsLength);
  }

  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


// static
SerializeResult Serializer::SerializeSTHSignatureInput(
    const ct::SignedTreeHead& sth, std::string* result) {
  CHECK(result);
  // TODO(alcutter): this should know whether it's V1 or V2 from the
  // Configure()
  switch (sth.version()) {
    case ct::V1:
      return SerializeV1STHSignatureInput(sth.timestamp(), sth.tree_size(),
                                          sth.sha256_root_hash(), result);
    case ct::V2:
      return SerializeV2STHSignatureInput(sth.timestamp(), sth.tree_size(),
                                          sth.sha256_root_hash(),
                                          sth.sth_extension(),
                                          sth.id().key_id(), result);
    default:
      break;
  }
  return SerializeResult::UNSUPPORTED_VERSION;
}


// static
SerializeResult Serializer::SerializeSCTMerkleTreeLeaf(
    const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
    std::string* result) {
  CHECK(result);
  CHECK(serialize_sct_merkle_leaf);
  return serialize_sct_merkle_leaf(sct, entry, result);
}


// static
SerializeResult Serializer::SerializeSCTSignatureInput(
    const SignedCertificateTimestamp& sct, const LogEntry& entry,
    string* result) {
  CHECK(result);
  CHECK(serialize_sct_sig_input);
  return serialize_sct_sig_input(sct, entry, result);
}


SerializeResult TLSSerializer::WriteSCTV1(
    const SignedCertificateTimestamp& sct) {
  CHECK(sct.version() == ct::V1);
  SerializeResult res = CheckExtensionsFormat(sct.extensions());
  if (res != SerializeResult::OK) {
    return res;
  }
  if (sct.id().key_id().size() != Serializer::kKeyIDLengthInBytes) {
    return SerializeResult::INVALID_KEYID_LENGTH;
  }
  WriteUint(sct.version(), Serializer::kVersionLengthInBytes);
  WriteFixedBytes(sct.id().key_id());
  WriteUint(sct.timestamp(), Serializer::kTimestampLengthInBytes);
  WriteVarBytes(sct.extensions(), Serializer::kMaxExtensionsLength);
  return WriteDigitallySigned(sct.signature());
}

SerializeResult TLSSerializer::WriteSCTV2(
    const SignedCertificateTimestamp& sct) {
  CHECK(sct.version() == ct::V2);
  SerializeResult res = CheckSctExtensionsFormat(sct.sct_extension());
  if (res != SerializeResult::OK) {
    return res;
  }
  if (sct.id().key_id().size() != Serializer::kKeyIDLengthInBytes) {
    return SerializeResult::INVALID_KEYID_LENGTH;
  }
  WriteUint(sct.version(), Serializer::kVersionLengthInBytes);
  WriteFixedBytes(sct.id().key_id());
  WriteUint(sct.timestamp(), Serializer::kTimestampLengthInBytes);
  // V2 SCT can have a number of extensions. They must be ordered by type
  // but we already checked that above.
  WriteSctExtension(sct.sct_extension());
  return WriteDigitallySigned(sct.signature());
}

// static
SerializeResult Serializer::SerializeSCT(const SignedCertificateTimestamp& sct,
                                         string* result) {
  TLSSerializer serializer;
  SerializeResult res = SerializeResult::UNSUPPORTED_VERSION;

  switch (sct.version()) {
    case ct::V1:
      res = serializer.WriteSCTV1(sct);
      break;
    case ct::V2:
      res = serializer.WriteSCTV2(sct);
      break;
    default:
      res = SerializeResult::UNSUPPORTED_VERSION;
  }
  if (res != SerializeResult::OK) {
    return res;
  }
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}

// static
SerializeResult Serializer::SerializeSCTList(
    const SignedCertificateTimestampList& sct_list, string* result) {
  if (sct_list.sct_list_size() == 0) {
    return SerializeResult::EMPTY_LIST;
  }
  return SerializeList(sct_list.sct_list(),
                       Serializer::kMaxSerializedSCTLength,
                       Serializer::kMaxSCTListLength, result);
}

// static
SerializeResult Serializer::SerializeDigitallySigned(
    const DigitallySigned& sig, string* result) {
  TLSSerializer serializer;
  SerializeResult res = serializer.WriteDigitallySigned(sig);
  if (res != SerializeResult::OK) {
    return res;
  }
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}

void TLSSerializer::WriteFixedBytes(const string& in) {
  output_.append(in);
}

void TLSSerializer::WriteVarBytes(const string& in, size_t max_length) {
  CHECK_LE(in.size(), max_length);

  size_t prefix_length = PrefixLength(max_length);
  WriteUint(in.size(), prefix_length);
  WriteFixedBytes(in);
}

// This does not enforce extension ordering, which must be done separately.
void TLSSerializer::WriteSctExtension(
    const RepeatedPtrField<SctExtension>& extension) {
  WriteUint(extension.size(), 2);
  for (auto it = extension.begin(); it != extension.end(); ++it) {
    WriteUint(it->sct_extension_type(), 2);
    WriteVarBytes(it->sct_extension_data(), Serializer::kMaxExtensionsLength);
  }
}


size_t SerializedListLength(const repeated_string& in, size_t max_elem_length,
                            size_t max_total_length) {
  size_t elem_prefix_length = PrefixLength(max_elem_length);
  size_t total_length = 0;

  for (int i = 0; i < in.size(); ++i) {
    if (in.Get(i).size() > max_elem_length ||
        max_total_length - total_length < elem_prefix_length ||
        max_total_length - total_length - elem_prefix_length <
            in.Get(i).size())
      return 0;

    total_length += elem_prefix_length + in.Get(i).size();
  }

  return total_length + PrefixLength(max_total_length);
}


// static
SerializeResult Serializer::SerializeList(const repeated_string& in,
                                          size_t max_elem_length,
                                          size_t max_total_length,
                                          string* result) {
  TLSSerializer serializer;
  SerializeResult res =
      serializer.WriteList(in, max_elem_length, max_total_length);
  if (res != SerializeResult::OK)
    return res;
  result->assign(serializer.SerializedString());
  return SerializeResult::OK;
}


SerializeResult TLSSerializer::WriteList(const repeated_string& in,
                                         size_t max_elem_length,
                                         size_t max_total_length) {
  for (int i = 0; i < in.size(); ++i) {
    if (in.Get(i).empty())
      return SerializeResult::EMPTY_ELEM_IN_LIST;
    if (in.Get(i).size() > max_elem_length)
      return SerializeResult::LIST_ELEM_TOO_LONG;
  }
  size_t length = SerializedListLength(in, max_elem_length, max_total_length);
  if (length == 0)
    return SerializeResult::LIST_TOO_LONG;
  size_t prefix_length = PrefixLength(max_total_length);
  CHECK_GE(length, prefix_length);

  WriteUint(length - prefix_length, prefix_length);

  for (int i = 0; i < in.size(); ++i)
    WriteVarBytes(in.Get(i), max_elem_length);
  return SerializeResult::OK;
}

SerializeResult TLSSerializer::WriteDigitallySigned(
    const DigitallySigned& sig) {
  SerializeResult res = CheckSignatureFormat(sig);
  if (res != SerializeResult::OK)
    return res;
  WriteUint(sig.hash_algorithm(), Serializer::kHashAlgorithmLengthInBytes);
  WriteUint(sig.sig_algorithm(), Serializer::kSigAlgorithmLengthInBytes);
  WriteVarBytes(sig.signature(), Serializer::kMaxSignatureLength);
  return SerializeResult::OK;
}


SerializeResult CheckKeyHashFormat(const string& key_hash) {
  if (key_hash.size() != Serializer::kKeyHashLengthInBytes)
    return SerializeResult::INVALID_HASH_LENGTH;
  return SerializeResult::OK;
}


SerializeResult CheckSignatureFormat(const DigitallySigned& sig) {
  // This is just DCHECKED upon setting, so check again.
  if (!DigitallySigned_HashAlgorithm_IsValid(sig.hash_algorithm()))
    return SerializeResult::INVALID_HASH_ALGORITHM;
  if (!DigitallySigned_SignatureAlgorithm_IsValid(sig.sig_algorithm()))
    return SerializeResult::INVALID_SIGNATURE_ALGORITHM;
  if (sig.signature().size() > Serializer::kMaxSignatureLength)
    return SerializeResult::SIGNATURE_TOO_LONG;
  return SerializeResult::OK;
}


SerializeResult CheckExtensionsFormat(const string& extensions) {
  if (extensions.size() > Serializer::kMaxExtensionsLength)
    return SerializeResult::EXTENSIONS_TOO_LONG;
  return SerializeResult::OK;
}


// Checks the (v2) STH extensions are correct. The RFC defines that there can
// be up to 65534 of them and each one can contain up to 65535 bytes.
// They must be in ascending order of extension type.
SerializeResult CheckSthExtensionsFormat(
    const repeated_sth_extension& extension) {
  if (extension.size() > Serializer::kMaxV2ExtensionsCount) {
    return SerializeResult::EXTENSIONS_TOO_LONG;
  }

  int32_t last_type_seen = 0;

  for (auto it = extension.begin(); it != extension.end(); ++it) {
    if (it->sth_extension_type() > Serializer::kMaxV2ExtensionType) {
      return SerializeResult::INVALID_ENTRY_TYPE;
    }

    if (it->sth_extension_data().size() > Serializer::kMaxExtensionsLength) {
      return SerializeResult::EXTENSIONS_TOO_LONG;
    }

    if (it->sth_extension_type() < last_type_seen) {
      // It's out of order - reject
      return SerializeResult::EXTENSIONS_NOT_ORDERED;
    }

    last_type_seen = it->sth_extension_type();
  }

  return SerializeResult::OK;
}


// Checks the (v2) SCT extensions are correct. The RFC defines that there can
// be up to 65534 of them and each one can contain up to 65535 bytes. They
// must be in ascending order of extension type
SerializeResult CheckSctExtensionsFormat(
    const RepeatedPtrField<SctExtension>& extension) {
  if (extension.size() > Serializer::kMaxV2ExtensionsCount) {
    return SerializeResult::EXTENSIONS_TOO_LONG;
  }

  int32_t last_type_seen = 0;

  for (auto it = extension.begin(); it != extension.end(); ++it) {
    if (it->sct_extension_data().size() > Serializer::kMaxExtensionsLength) {
      return SerializeResult::EXTENSIONS_TOO_LONG;
    }

    if (it->sct_extension_type() > Serializer::kMaxV2ExtensionType) {
      return SerializeResult::INVALID_ENTRY_TYPE;
    }

    if (it->sct_extension_type() < last_type_seen) {
      // It's out of order - reject
      return SerializeResult::EXTENSIONS_NOT_ORDERED;
    }

    last_type_seen = it->sct_extension_type();
  }

  return SerializeResult::OK;
}


const size_t TLSDeserializer::kV2ExtensionCountLengthInBytes = 2;
const size_t TLSDeserializer::kV2ExtensionTypeLengthInBytes = 2;


TLSDeserializer::TLSDeserializer(const string& input)
    : current_pos_(input.data()), bytes_remaining_(input.size()) {
}


DeserializeResult TLSDeserializer::ReadSCTV1(SignedCertificateTimestamp* sct) {
  sct->set_version(ct::V1);
  if (!ReadFixedBytes(Serializer::kKeyIDLengthInBytes,
                      sct->mutable_id()->mutable_key_id())) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  // V1 encoding.
  uint64_t timestamp = 0;
  if (!ReadUint(Serializer::kTimestampLengthInBytes, &timestamp)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  sct->set_timestamp(timestamp);
  string extensions;
  if (!ReadVarBytes(Serializer::kMaxExtensionsLength, &extensions)) {
    // In theory, could also be an invalid length prefix, but not if
    // length limits follow byte boundaries.
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  return ReadDigitallySigned(sct->mutable_signature());
}

DeserializeResult TLSDeserializer::ReadSCTV2(SignedCertificateTimestamp* sct) {
  sct->set_version(ct::V2);
  if (!ReadFixedBytes(Serializer::kKeyIDLengthInBytes,
                      sct->mutable_id()->mutable_key_id())) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  // V2 encoding.
  uint64_t timestamp = 0;
  if (!ReadUint(Serializer::kTimestampLengthInBytes, &timestamp)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  sct->set_timestamp(timestamp);
  // Extensions are handled differently for V2
  const DeserializeResult res = ReadSctExtension(sct->mutable_sct_extension());
  if (res != DeserializeResult::OK) {
    return res;
  }
  return ReadDigitallySigned(sct->mutable_signature());
}


DeserializeResult TLSDeserializer::ReadSCT(SignedCertificateTimestamp* sct) {
  int version;
  if (!ReadUint(Serializer::kVersionLengthInBytes, &version)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  if (!Version_IsValid(version) || (version != ct::V1 && version != ct::V2)) {
    return DeserializeResult::UNSUPPORTED_VERSION;
  }

  switch (version) {
    case ct::V1:
      return ReadSCTV1(sct);
      break;

    case ct::V2:

      return ReadSCTV2(sct);
      break;

    default:
      return DeserializeResult::UNSUPPORTED_VERSION;
  }
}


// static
DeserializeResult Deserializer::DeserializeSCT(
    const string& in, SignedCertificateTimestamp* sct) {
  TLSDeserializer deserializer(in);
  DeserializeResult res = deserializer.ReadSCT(sct);
  if (res != DeserializeResult::OK) {
    return res;
  }
  if (!deserializer.ReachedEnd()) {
    return DeserializeResult::INPUT_TOO_LONG;
  }
  return DeserializeResult::OK;
}


// static
DeserializeResult Deserializer::DeserializeSCTList(
    const string& in, SignedCertificateTimestampList* sct_list) {
  sct_list->clear_sct_list();
  DeserializeResult res = DeserializeList(in, Serializer::kMaxSCTListLength,
                                          Serializer::kMaxSerializedSCTLength,
                                          sct_list->mutable_sct_list());
  if (res != DeserializeResult::OK)
    return res;
  if (sct_list->sct_list_size() == 0)
    return DeserializeResult::EMPTY_LIST;
  return DeserializeResult::OK;
}


// static
DeserializeResult Deserializer::DeserializeDigitallySigned(
    const string& in, DigitallySigned* sig) {
  TLSDeserializer deserializer(in);
  DeserializeResult res = deserializer.ReadDigitallySigned(sig);
  if (res != DeserializeResult::OK)
    return res;
  if (!deserializer.ReachedEnd())
    return DeserializeResult::INPUT_TOO_LONG;
  return DeserializeResult::OK;
}


bool TLSDeserializer::ReadFixedBytes(size_t bytes, string* result) {
  if (bytes_remaining_ < bytes)
    return false;
  result->assign(current_pos_, bytes);
  current_pos_ += bytes;
  bytes_remaining_ -= bytes;
  return true;
}


bool TLSDeserializer::ReadLengthPrefix(size_t max_length, size_t* result) {
  size_t prefix_length = PrefixLength(max_length);
  size_t length;
  if (!ReadUint(prefix_length, &length) || length > max_length)
    return false;
  *result = length;
  return true;
}


bool TLSDeserializer::ReadVarBytes(size_t max_length, string* result) {
  size_t length;
  if (!ReadLengthPrefix(max_length, &length))
    return false;
  return ReadFixedBytes(length, result);
}


// static
DeserializeResult Deserializer::DeserializeList(const string& in,
                                                size_t max_total_length,
                                                size_t max_elem_length,
                                                repeated_string* out) {
  TLSDeserializer deserializer(in);
  DeserializeResult res =
      deserializer.ReadList(max_total_length, max_elem_length, out);
  if (res != DeserializeResult::OK)
    return res;
  if (!deserializer.ReachedEnd())
    return DeserializeResult::INPUT_TOO_LONG;
  return DeserializeResult::OK;
}


DeserializeResult TLSDeserializer::ReadList(size_t max_total_length,
                                            size_t max_elem_length,
                                            repeated_string* out) {
  string serialized_list;
  if (!ReadVarBytes(max_total_length, &serialized_list))
    // TODO(ekasper): could also be a length that's too large, if
    // length limits don't follow byte boundaries.
    return DeserializeResult::INPUT_TOO_SHORT;
  if (!ReachedEnd())
    return DeserializeResult::INPUT_TOO_LONG;

  TLSDeserializer list_reader(serialized_list);
  while (!list_reader.ReachedEnd()) {
    string elem;
    if (!list_reader.ReadVarBytes(max_elem_length, &elem))
      return DeserializeResult::INVALID_LIST_ENCODING;
    if (elem.empty())
      return DeserializeResult::EMPTY_ELEM_IN_LIST;
    *(out->Add()) = elem;
  }
  return DeserializeResult::OK;
}


DeserializeResult TLSDeserializer::ReadSctExtension(
    RepeatedPtrField<SctExtension>* extension) {
  uint32_t ext_count;
  if (!ReadUint(kV2ExtensionCountLengthInBytes, &ext_count)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }

  if (ext_count > Serializer::kMaxV2ExtensionsCount) {
    return DeserializeResult::EXTENSIONS_TOO_LONG;
  }

  for (int ext = 0; ext < ext_count; ++ext) {
    uint32_t ext_type;
    if (!ReadUint(kV2ExtensionTypeLengthInBytes, &ext_type)) {
      return DeserializeResult::INPUT_TOO_SHORT;
    }

    string ext_data;
    if (!ReadVarBytes(Serializer::kMaxExtensionsLength, &ext_data)) {
      return DeserializeResult::INPUT_TOO_SHORT;
    }

    SctExtension* new_ext = extension->Add();
    new_ext->set_sct_extension_type(ext_type);
    new_ext->set_sct_extension_data(ext_data);
  }

  // This makes sure they're correctly ordered (See RFC section 5.3)
  return CheckSctExtensionsFormat(*extension) == SerializeResult::OK
             ? DeserializeResult::OK
             : DeserializeResult::EXTENSIONS_NOT_ORDERED;
}


DeserializeResult TLSDeserializer::ReadDigitallySigned(DigitallySigned* sig) {
  int hash_algo = -1, sig_algo = -1;
  if (!ReadUint(Serializer::kHashAlgorithmLengthInBytes, &hash_algo))
    return DeserializeResult::INPUT_TOO_SHORT;
  if (!DigitallySigned_HashAlgorithm_IsValid(hash_algo))
    return DeserializeResult::INVALID_HASH_ALGORITHM;
  if (!ReadUint(Serializer::kSigAlgorithmLengthInBytes, &sig_algo))
    return DeserializeResult::INPUT_TOO_SHORT;
  if (!DigitallySigned_SignatureAlgorithm_IsValid(sig_algo))
    return DeserializeResult::INVALID_SIGNATURE_ALGORITHM;

  string sig_string;
  if (!ReadVarBytes(Serializer::kMaxSignatureLength, &sig_string))
    return DeserializeResult::INPUT_TOO_SHORT;
  sig->set_hash_algorithm(
      static_cast<DigitallySigned::HashAlgorithm>(hash_algo));
  sig->set_sig_algorithm(
      static_cast<DigitallySigned::SignatureAlgorithm>(sig_algo));
  sig->set_signature(sig_string);
  return DeserializeResult::OK;
}


DeserializeResult TLSDeserializer::ReadExtensions(
    ct::TimestampedEntry* entry) {
  string extensions;
  if (!ReadVarBytes(Serializer::kMaxExtensionsLength, &extensions)) {
    return DeserializeResult::INPUT_TOO_SHORT;
  }
  CHECK_NOTNULL(entry)->set_extensions(extensions);
  return DeserializeResult::OK;
}


DeserializeResult Deserializer::DeserializeMerkleTreeLeaf(
    const std::string& in, ct::MerkleTreeLeaf* leaf) {
  TLSDeserializer des(in);

  DeserializeResult ret = read_merkle_tree_leaf(&des, leaf);
  if (ret != DeserializeResult::OK) {
    return ret;
  }

  if (!des.ReachedEnd()) {
    return DeserializeResult::INPUT_TOO_LONG;
  }

  return DeserializeResult::OK;
}


// static
void Serializer::ConfigureV1(
    const function<string(const ct::LogEntry&)>& leaf_data_func,
    const function<SerializeResult(
        const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
        std::string* result)>& serialize_sct_sig_input_func,
    const function<SerializeResult(
        const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
        std::string* result)>& serialize_sct_merkle_leaf_func) {
  CHECK(FLAGS_allow_reconfigure_serializer_test_only ||
        (!leaf_data&& !serialize_sct_sig_input &&
         !serialize_sct_merkle_leaf))
      << "Serializer already configured";
  leaf_data = leaf_data_func;
  serialize_sct_sig_input = serialize_sct_sig_input_func;
  serialize_sct_merkle_leaf = serialize_sct_merkle_leaf_func;
  serialize_sth_sig_input_v1 = ::SerializeV1STHSignatureInput;
}


// static
void Serializer::ConfigureV2(
    const function<string(const ct::LogEntry&)>& leaf_data_func,
    const function<SerializeResult(
        const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
        std::string* result)>& serialize_sct_sig_input_func,
    const function<SerializeResult(
        const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
        std::string* result)>& serialize_sct_merkle_leaf_func) {
  CHECK(FLAGS_allow_reconfigure_serializer_test_only ||
        (!leaf_data && !serialize_sct_sig_input &&
         !serialize_sct_merkle_leaf))
      << "Serializer already configured";
  leaf_data = leaf_data_func;
  serialize_sct_sig_input = serialize_sct_sig_input_func;
  serialize_sct_merkle_leaf = serialize_sct_merkle_leaf_func;
  serialize_sth_sig_input_v2 = ::SerializeV2STHSignatureInput;
}


// static
void Deserializer::Configure(
    const function<DeserializeResult(TLSDeserializer* d,
                                     ct::MerkleTreeLeaf* leaf)>&
        read_merkle_tree_leaf_func) {
  CHECK(FLAGS_allow_reconfigure_serializer_test_only ||
        !read_merkle_tree_leaf)
      << "Deserializer already configured";
  read_merkle_tree_leaf = read_merkle_tree_leaf_func;
}
