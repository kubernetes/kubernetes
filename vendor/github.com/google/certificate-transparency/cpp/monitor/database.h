#ifndef MONITOR_DATABASE_H
#define MONITOR_DATABASE_H

#include <glog/logging.h>
#include <stdint.h>

#include "base/macros.h"
#include "log/logged_entry.h"

namespace monitor {

class Database {
 public:
  enum WriteResult {
    WRITE_OK,
    SERIALIZE_FAILED,
    DUPLICATE_TIMESTAMP,
    NOT_ALLOWED,
    WRITE_FAILED,
  };

  enum LookupResult {
    LOOKUP_OK,
    NOT_FOUND,
  };

  enum VerificationLevel {
    SIGNATURE_VERIFICATION_FAILED,
    SIGNATURE_VERIFIED,
    TREE_CONFIRMED,
    TREE_CONFIRMATION_FAILED,
    INCONSISTENT,  // Good signature but not sane (e.g. timestamp in the
                   // future).
    UNDEFINED,     // Let this be last.
  };

  Database() = default;
  virtual ~Database() = default;

  virtual void BeginTransaction() {
    DLOG(FATAL) << "Transactions not supported";
  }

  virtual void EndTransaction() {
    DLOG(FATAL) << "Transactions not supported";
  }

  static std::string VerificationLevelString(VerificationLevel result) {
    switch (result) {
      case SIGNATURE_VERIFICATION_FAILED:
        return "Signature verification failed.";
      case SIGNATURE_VERIFIED:
        return "Signature verified.";
      case TREE_CONFIRMED:
        return "Tree Confirmed.";
      case TREE_CONFIRMATION_FAILED:
        return "Tree confirmation failed.";
      case INCONSISTENT:
        return "Signature verified but inconsistent STH.";
      case UNDEFINED:
        return "STH not yet verified.";
      default:
        LOG(FATAL) << "unknown VerificationLevel enum value: " << result;
        return "";
    }
  }

  // Attempt to create a new entry. The caller has to ensure
  // everything itself (i.e. no UNIQUE constraints).  Do preprocessing
  // here independent from database implementation.
  // cert_trans::LoggedEntry is here only for being a container
  // for SignedCertificateTimestamp and LogEntry built in
  // GetEntries().  The latter two contain all information from the
  // RFC compliant get-entries response from the log server.
  WriteResult CreateEntry(const cert_trans::LoggedEntry& logged);

  virtual WriteResult WriteSTH(const ct::SignedTreeHead& sth);

  // Lookup latest *written* STH (i.e. not necessarily latest timestamp).
  virtual LookupResult LookupLatestWrittenSTH(
      ct::SignedTreeHead* result) const = 0;

  virtual LookupResult LookupHashByIndex(int64_t sequence_number,
                                         std::string* result) const = 0;

  virtual WriteResult SetVerificationLevel(const ct::SignedTreeHead& sth,
                                           VerificationLevel verify_level);

  virtual LookupResult LookupSTHByTimestamp(
      uint64_t timestamp, ct::SignedTreeHead* result) const = 0;

  virtual LookupResult LookupVerificationLevel(
      const ct::SignedTreeHead& sth, VerificationLevel* result) const = 0;

 private:
  virtual WriteResult CreateEntry_(const std::string& leaf,
                                   const std::string& leaf_hash,
                                   const std::string& cert,
                                   const std::string& cert_chain) = 0;

  virtual WriteResult WriteSTH_(uint64_t timestamp, int64_t tree_size,
                                const std::string& sth) = 0;

  virtual WriteResult SetVerificationLevel_(
      const ct::SignedTreeHead& sth, VerificationLevel verify_level) = 0;

  DISALLOW_COPY_AND_ASSIGN(Database);
};

}  // namespace monitor

#endif  // MONITOR_DATABASE_H
