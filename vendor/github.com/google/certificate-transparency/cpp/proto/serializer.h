/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <glog/logging.h>
#include <google/protobuf/repeated_field.h>
#include <stdint.h>
#include <functional>
#include <string>

#include "base/macros.h"
#include "proto/ct.pb.h"


// Serialization methods return OK on success,
// or the first encountered error on failure.
enum class SerializeResult {
  OK,
  INVALID_ENTRY_TYPE,
  EMPTY_CERTIFICATE,
  // TODO(alcutter): rename these to LEAFDATA_TOO_LONG or similar?
  CERTIFICATE_TOO_LONG,
  CERTIFICATE_CHAIN_TOO_LONG,
  INVALID_HASH_ALGORITHM,
  INVALID_SIGNATURE_ALGORITHM,
  SIGNATURE_TOO_LONG,
  INVALID_HASH_LENGTH,
  EMPTY_PRECERTIFICATE_CHAIN,
  UNSUPPORTED_VERSION,
  EXTENSIONS_TOO_LONG,
  INVALID_KEYID_LENGTH,
  EMPTY_LIST,
  EMPTY_ELEM_IN_LIST,
  LIST_ELEM_TOO_LONG,
  LIST_TOO_LONG,
  EXTENSIONS_NOT_ORDERED,
};

std::ostream& operator<<(std::ostream& stream, const SerializeResult& r);


enum class DeserializeResult {
  OK,
  INPUT_TOO_SHORT,
  INVALID_HASH_ALGORITHM,
  INVALID_SIGNATURE_ALGORITHM,
  INPUT_TOO_LONG,
  UNSUPPORTED_VERSION,
  INVALID_LIST_ENCODING,
  EMPTY_LIST,
  EMPTY_ELEM_IN_LIST,
  UNKNOWN_LEAF_TYPE,
  UNKNOWN_LOGENTRY_TYPE,
  EXTENSIONS_TOO_LONG,
  EXTENSIONS_NOT_ORDERED,
};

std::ostream& operator<<(std::ostream& stream, const DeserializeResult& r);

typedef google::protobuf::RepeatedPtrField<std::string> repeated_string;
typedef google::protobuf::RepeatedPtrField<ct::SthExtension>
    repeated_sth_extension;
typedef google::protobuf::RepeatedPtrField<ct::SctExtension>
    repeated_sct_extension;

SerializeResult CheckExtensionsFormat(const std::string& extensions);
SerializeResult CheckKeyHashFormat(const std::string& key_hash);
SerializeResult CheckSctExtensionsFormat(
    const repeated_sct_extension& extension);

// TODO(pphaneuf): Make this into normal functions in a namespace.
class TLSSerializer {
 public:
  // returns binary data
  std::string SerializedString() const {
    return output_;
  }

  SerializeResult WriteSCTV1(const ct::SignedCertificateTimestamp& sct);

  SerializeResult WriteSCTV2(const ct::SignedCertificateTimestamp& sct);

  SerializeResult WriteList(const repeated_string& in, size_t max_elem_length,
                            size_t max_total_length);

  SerializeResult WriteDigitallySigned(const ct::DigitallySigned& sig);


  template <class T>
  void WriteUint(T in, size_t bytes) {
    CHECK_LE(bytes, sizeof(in));
    CHECK(bytes == sizeof(in) || in >> (bytes * 8) == 0);
    for (; bytes > 0; --bytes)
      output_.push_back(((in & (static_cast<T>(0xff) << ((bytes - 1) * 8))) >>
                         ((bytes - 1) * 8)));
  }

  // Fixed-length byte array.
  void WriteFixedBytes(const std::string& in);

  // Variable-length byte array.
  // Caller is responsible for checking |in| <= max_length
  // TODO(ekasper): could return a bool instead.
  void WriteVarBytes(const std::string& in, size_t max_length);

  void WriteSctExtension(const repeated_sct_extension& extension);

 private:
  std::string output_;
};


class TLSDeserializer {
 public:
  // We do not make a copy, so input must remain valid.
  // TODO(pphaneuf): And so we should take a string *, not a string &
  // (which could be to a temporary, and not valid once the
  // constructor returns).
  explicit TLSDeserializer(const std::string& input);

  bool ReachedEnd() const {
    return bytes_remaining_ == 0;
  }

  DeserializeResult ReadSCT(ct::SignedCertificateTimestamp* sct);

  DeserializeResult ReadList(size_t max_total_length, size_t max_elem_length,
                             repeated_string* out);

  DeserializeResult ReadDigitallySigned(ct::DigitallySigned* sig);

  DeserializeResult ReadMerkleTreeLeaf(ct::MerkleTreeLeaf* leaf);
  bool ReadVarBytes(size_t max_length, std::string* result);

  template <class T>
  bool ReadUint(size_t bytes, T* result) {
    if (bytes_remaining_ < bytes)
      return false;
    T res = 0;
    for (size_t i = 0; i < bytes; ++i) {
      res = (res << 8) | static_cast<unsigned char>(*current_pos_);
      ++current_pos_;
    }

    bytes_remaining_ -= bytes;
    *result = res;
    return true;
  }

  DeserializeResult ReadExtensions(ct::TimestampedEntry* entry);
  bool ReadFixedBytes(size_t bytes, std::string* result);

 private:
  static const size_t kV2ExtensionCountLengthInBytes;
  static const size_t kV2ExtensionTypeLengthInBytes;

  DeserializeResult ReadSctExtension(repeated_sct_extension* extension);
  DeserializeResult ReadMerkleTreeLeafV1(ct::MerkleTreeLeaf* leaf);
  DeserializeResult ReadMerkleTreeLeafV2(ct::MerkleTreeLeaf* leaf);
  DeserializeResult ReadSCTV1(ct::SignedCertificateTimestamp* sct);
  DeserializeResult ReadSCTV2(ct::SignedCertificateTimestamp* sct);
  bool ReadLengthPrefix(size_t max_length, size_t* result);

  const char* current_pos_;
  size_t bytes_remaining_;

  DISALLOW_COPY_AND_ASSIGN(TLSDeserializer);
};

// A utility class for writing protocol buffer fields in canonical TLS style.
class Serializer {
 public:
  static const size_t kMaxSignatureLength;
  static const size_t kMaxV2ExtensionType;
  static const size_t kMaxV2ExtensionsCount;
  static const size_t kMaxExtensionsLength;
  static const size_t kMaxSerializedSCTLength;
  static const size_t kMaxSCTListLength;

  static const size_t kLogEntryTypeLengthInBytes;
  static const size_t kSignatureTypeLengthInBytes;
  static const size_t kHashAlgorithmLengthInBytes;
  static const size_t kSigAlgorithmLengthInBytes;
  static const size_t kVersionLengthInBytes;
  // Log Key ID
  static const size_t kKeyIDLengthInBytes;
  static const size_t kMerkleLeafTypeLengthInBytes;
  // Public key hash from cert
  static const size_t kKeyHashLengthInBytes;
  static const size_t kTimestampLengthInBytes;

  // API
  // TODO(alcutter): typedef these function<> bits
  static void ConfigureV1(
      const std::function<std::string(const ct::LogEntry&)>& leaf_data,
      const std::function<SerializeResult(
          const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
          std::string* result)>& serialize_sct_sig_input,
      const std::function<SerializeResult(
          const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
          std::string* result)>& serialize_sct_merkle_leaf);

  static void ConfigureV2(
      const std::function<std::string(const ct::LogEntry&)>& leaf_data,
      const std::function<SerializeResult(
          const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
          std::string* result)>& serialize_sct_sig_input,
      const std::function<SerializeResult(
          const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
          std::string* result)>& serialize_sct_merkle_leaf);

  static std::string LeafData(const ct::LogEntry& entry);

  static SerializeResult SerializeSTHSignatureInput(
      const ct::SignedTreeHead& sth, std::string* result);

  static SerializeResult SerializeSCTMerkleTreeLeaf(
      const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
      std::string* result);

  static SerializeResult SerializeSCTSignatureInput(
      const ct::SignedCertificateTimestamp& sct, const ct::LogEntry& entry,
      std::string* result);

  static SerializeResult SerializeV1STHSignatureInput(
      uint64_t timestamp, int64_t tree_size, const std::string& root_hash,
      std::string* result);

  static SerializeResult SerializeV2STHSignatureInput(
      uint64_t timestamp, int64_t tree_size, const std::string& root_hash,
      const repeated_sth_extension& sth_extension, const std::string& log_id,
      std::string* result);


  // Random utils
  static SerializeResult SerializeList(const repeated_string& in,
                                       size_t max_elem_length,
                                       size_t max_total_length,
                                       std::string* result);

  static SerializeResult SerializeSCT(
      const ct::SignedCertificateTimestamp& sct, std::string* result);

  static SerializeResult SerializeSCTList(
      const ct::SignedCertificateTimestampList& sct_list, std::string* result);

  static SerializeResult SerializeDigitallySigned(
      const ct::DigitallySigned& sig, std::string* result);

  // TODO(ekasper): tests for these!
  template <class T>
  static std::string SerializeUint(T in, size_t bytes = sizeof(T)) {
    TLSSerializer serializer;
    serializer.WriteUint(in, bytes);
    return serializer.SerializedString();
  }

 private:
  // This class is mostly a namespace for static methods.
  // TODO(pphaneuf): Make this into normal functions in a namespace.
  Serializer() = delete;
};


class Deserializer {
 public:
  static void Configure(const std::function<DeserializeResult(
                            TLSDeserializer* d, ct::MerkleTreeLeaf* leaf)>&
                            read_merkle_tree_leaf_body);

  static DeserializeResult DeserializeSCT(const std::string& in,
                                          ct::SignedCertificateTimestamp* sct);

  static DeserializeResult DeserializeSCTList(
      const std::string& in, ct::SignedCertificateTimestampList* sct_list);

  static DeserializeResult DeserializeDigitallySigned(
      const std::string& in, ct::DigitallySigned* sig);

  // FIXME(ekasper): for simplicity these reject if the list has empty
  // elements (all our use cases are like this) but they should take in
  // an arbitrary min bound instead.
  static DeserializeResult DeserializeList(const std::string& in,
                                           size_t max_total_length,
                                           size_t max_elem_length,
                                           repeated_string* out);

  static DeserializeResult DeserializeMerkleTreeLeaf(const std::string& in,
                                                     ct::MerkleTreeLeaf* leaf);

  // TODO(pphaneuf): Maybe the users of this should just use
  // TLSDeserializer directly?
  template <class T>
  static DeserializeResult DeserializeUint(const std::string& in, size_t bytes,
                                           T* result) {
    TLSDeserializer deserializer(in);
    bool res = deserializer.ReadUint(bytes, result);
    if (!res)
      return DeserializeResult::INPUT_TOO_SHORT;
    if (!deserializer.ReachedEnd())
      return DeserializeResult::INPUT_TOO_LONG;
    return DeserializeResult::OK;
  }

 private:
  // This class is mostly a namespace for static methods.
  // TODO(pphaneuf): Make this into normal functions in a namespace.
  Deserializer() = delete;

  // This should never do anything, but just in case...
  DISALLOW_COPY_AND_ASSIGN(Deserializer);
};

#endif
