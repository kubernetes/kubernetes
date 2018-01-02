/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef LOG_VERIFIER_H
#define LOG_VERIFIER_H

#include <glog/logging.h>
#include <stdint.h>

#include "base/macros.h"
#include "log/log_signer.h"
#include "proto/ct.pb.h"

class MerkleVerifier;

// A verifier for verifying signed statements of the log.
// TODO(ekasper): unit tests.
class LogVerifier {
 public:
  LogVerifier(LogSigVerifier* sig_verifier, MerkleVerifier* tree_verifier);
  ~LogVerifier();

  enum LogVerifyResult {
    VERIFY_OK,
    // Invalid format
    INVALID_FORMAT,
    // Invalid timestamp
    INVALID_TIMESTAMP,
    // Timestamps differ.
    INCONSISTENT_TIMESTAMPS,
    // The signature does not verify.
    INVALID_SIGNATURE,
    // The Merkle path is not valid for the given index
    // and tree size
    INVALID_MERKLE_PATH,
  };

  static std::string VerifyResultString(LogVerifyResult result) {
    switch (result) {
      case VERIFY_OK:
        return "Verify OK.";
      case INVALID_FORMAT:
        return "Invalid format.";
      case INVALID_TIMESTAMP:
        return "Invalid timestamp.";
      case INCONSISTENT_TIMESTAMPS:
        return "Inconsistent timestamps.";
      case INVALID_SIGNATURE:
        return "Invalid signature.";
      case INVALID_MERKLE_PATH:
        return "Invalid Merkle path.";
    }

    LOG(FATAL) << "unknown value for LogVerifyResult enum: " << result;
    // It should not be possible for control to reach this point but this
    // avoids a compilation warning on some platforms.
    abort();
  }

  std::string KeyID() const {
    return sig_verifier_->KeyID();
  }

  // Verify that the timestamp is in the given range,
  // and the signature is valid.
  // Timestamps are given in milliseconds, since January 1, 1970,
  // 00:00 UTC time.
  LogVerifyResult VerifySignedCertificateTimestamp(
      const ct::LogEntry& entry, const ct::SignedCertificateTimestamp& sct,
      uint64_t begin_range, uint64_t end_range,
      std::string* merkle_leaf_hash) const;

  // Verify that the timestamp is not in the future, and the signature is
  // valid.
  LogVerifyResult VerifySignedCertificateTimestamp(
      const ct::LogEntry& entry, const ct::SignedCertificateTimestamp& sct,
      std::string* merkle_leaf_hash) const;

  LogVerifyResult VerifySignedCertificateTimestamp(
      const ct::LogEntry& entry, const ct::SignedCertificateTimestamp& sct,
      uint64_t begin_range, uint64_t end_range) const {
    return VerifySignedCertificateTimestamp(entry, sct, begin_range, end_range,
                                            NULL);
  }
  // Verify that the timestamp is not in the future, and the signature is
  // valid.
  LogVerifyResult VerifySignedCertificateTimestamp(
      const ct::LogEntry& entry,
      const ct::SignedCertificateTimestamp& sct) const {
    return VerifySignedCertificateTimestamp(entry, sct, NULL);
  }

  // Verify that the timestamp is in the given range,
  // and the signature is valid.
  // Timestamps are given in milliseconds, since January 1, 1970,
  // 00:00 UTC time.
  LogVerifyResult VerifySignedTreeHead(const ct::SignedTreeHead& sth,
                                       uint64_t begin_range,
                                       uint64_t end_range) const;

  // Verify that the timestamp is not in the future, and the signature is
  // valid.
  LogVerifyResult VerifySignedTreeHead(const ct::SignedTreeHead& sth) const;

  // Verify that
  // (1) The audit proof signature is valid
  // (2) sct timestamp <= audit proof timestamp <= now
  // Does not verify the SCT signature (or even require one).
  LogVerifyResult VerifyMerkleAuditProof(
      const ct::LogEntry& entry, const ct::SignedCertificateTimestamp& sct,
      const ct::MerkleAuditProof& merkle_proof) const;

  bool VerifyConsistency(const ct::SignedTreeHead& sth1,
                         const ct::SignedTreeHead& sth2,
                         const std::vector<std::string>& proof) const;

 private:
  LogSigVerifier* sig_verifier_;
  MerkleVerifier* merkle_verifier_;

  static bool IsBetween(uint64_t timestamp, uint64_t earliest,
                        uint64_t latest);

  DISALLOW_COPY_AND_ASSIGN(LogVerifier);
};
#endif
