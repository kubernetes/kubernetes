/* -*- mode: c++; indent-tabs-mode: nil -*- */

#ifndef LOGGED_ENTRY_H
#define LOGGED_ENTRY_H

#include <glog/logging.h>

#include "client/async_log_client.h"
#include "merkletree/serial_hasher.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/util.h"

namespace cert_trans {

class LoggedEntry : public ct::LoggedEntryPB {
 public:
  std::string Hash() const {
    return Sha256Hasher::Sha256Digest(Serializer::LeafData(entry()));
  }

  uint64_t timestamp() const {
    return sct().timestamp();
  }

  const ct::SignedCertificateTimestamp& sct() const {
    return contents().sct();
  }

  ct::SignedCertificateTimestamp* mutable_sct() {
    return mutable_contents()->mutable_sct();
  }

  const ct::LogEntry& entry() const {
    return contents().entry();
  }

  ct::LogEntry* mutable_entry() {
    return mutable_contents()->mutable_entry();
  }

  bool SerializeForDatabase(std::string* dst) const {
    return contents().SerializeToString(dst);
  }

  bool ParseFromDatabase(const std::string& src) {
    return mutable_contents()->ParseFromString(src);
  }

  bool SerializeForLeaf(std::string* dst) const {
    return Serializer::SerializeSCTMerkleTreeLeaf(sct(), entry(), dst) ==
           SerializeResult::OK;
  }

  bool SerializeExtraData(std::string* dst) const {
    switch (entry().type()) {
      case ct::X509_ENTRY:
        return SerializeX509Chain(entry().x509_entry(), dst) ==
               SerializeResult::OK;
      case ct::PRECERT_ENTRY:
        return SerializePrecertChainEntry(entry().precert_entry(), dst) ==
               SerializeResult::OK;
      case ct::PRECERT_ENTRY_V2:
        // TODO(mhs): V2 implementation needs to be provided.
        LOG(FATAL) << "CT V2 not yet implemented";
        break;
      case ct::X_JSON_ENTRY:
        dst->clear();
        return true;
      case ct::UNKNOWN_ENTRY_TYPE:
        // We'll handle this below, along with any unknown unknown types too.
        break;
    }
    LOG(FATAL) << "Unknown entry type " << entry().type();
  }

  // Note that this method will not fully populate the SCT.
  bool CopyFromClientLogEntry(const AsyncLogClient::Entry& entry);

  // FIXME(benl): unify with TestSigner?
  void RandomForTest() {
    const char kKeyID[] =
        "b69d879e3f2c4402556dcda2f6b2e02ff6b6df4789c53000e14f4b125ae847aa";

    mutable_sct()->set_version(ct::V1);
    mutable_sct()->mutable_id()->set_key_id(util::BinaryString(kKeyID));
    mutable_sct()->set_timestamp(util::TimeInMilliseconds());
    mutable_sct()->clear_extensions();

    int random_bits = rand();
    ct::LogEntryType type =
        random_bits & 1 ? ct::X509_ENTRY : ct::PRECERT_ENTRY;

    ct::LogEntry* entry = mutable_entry();

    entry->set_type(type);
    entry->clear_x509_entry();
    entry->clear_precert_entry();

    if (type == ct::X509_ENTRY) {
      entry->mutable_x509_entry()->set_leaf_certificate(
          util::RandomString(512, 1024));
      if (random_bits & 2) {
        entry->mutable_x509_entry()->add_certificate_chain(
            util::RandomString(512, 1024));

        if (random_bits & 4) {
          entry->mutable_x509_entry()->add_certificate_chain(
              util::RandomString(512, 1024));
        }
      }
    } else {
      entry->mutable_precert_entry()->mutable_pre_cert()->set_issuer_key_hash(
          util::RandomString(32, 32));
      entry->mutable_precert_entry()->mutable_pre_cert()->set_tbs_certificate(
          util::RandomString(512, 1024));
      entry->mutable_precert_entry()->set_pre_certificate(
          util::RandomString(512, 1024));
      if (random_bits & 2) {
        entry->mutable_precert_entry()->add_precertificate_chain(
            util::RandomString(512, 1024));

        if (random_bits & 4) {
          entry->mutable_precert_entry()->add_precertificate_chain(
              util::RandomString(512, 1024));
        }
      }
    }
  }
};


inline bool operator==(const LoggedEntry& lhs, const LoggedEntry& rhs) {
  // TODO(alcutter): Do this properly
  std::string l_str, r_str;
  CHECK(lhs.SerializeToString(&l_str));
  CHECK(rhs.SerializeToString(&r_str));
  return l_str == r_str;
}


inline bool operator==(const ct::LogEntry& lhs, const ct::LogEntry& rhs) {
  // TODO(alcutter): Do this properly
  std::string l_str, r_str;
  CHECK(lhs.SerializeToString(&l_str));
  CHECK(rhs.SerializeToString(&r_str));
  return l_str == r_str;
}


inline bool operator==(const ct::SignedCertificateTimestamp& lhs,
                       const ct::SignedCertificateTimestamp& rhs) {
  // TODO(alcutter): Do this properly
  std::string l_str, r_str;
  CHECK(lhs.SerializeToString(&l_str));
  CHECK(rhs.SerializeToString(&r_str));
  return l_str == r_str;
}


}  // namespace cert_trans

#endif  // LOGGED_ENTRY_H
