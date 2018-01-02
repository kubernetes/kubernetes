#ifndef MONITOR_H
#define MONITOR_H

#include <stdint.h>

#include "base/macros.h"

class LogVerifier;

namespace ct {
class SignedTreeHead;
}

namespace cert_trans {
class HTTPLogClient;
}

namespace monitor {

class Database;

class Monitor {
 public:
  enum GetResult {
    OK = 0,
    NETWORK_PROBLEM = 1,
  };

  enum VerifyResult {
    SIGNATURE_VALID = 0,
    SIGNATURE_INVALID = 1,
    // The STH is malformed (i.e. timestamp in the future)
    // but its signature is valid.
    STH_MALFORMED_WTH_VALID_SIGNATURE = 2,
  };

  enum ConfirmResult {
    TREE_CONFIRMED = 0,
    TREE_CONFIRMATION_FAILED = 1,
  };

  Monitor(Database* database, LogVerifier* verifier,
          cert_trans::HTTPLogClient* client, uint64_t sleep_time_sec);

  GetResult GetSTH();

  VerifyResult VerifySTH(uint64_t timestamp);

  GetResult GetEntries(int get_first, int get_last);

  ConfirmResult ConfirmTree(uint64_t timestamp);

  void Init();

  void Loop();

 private:
  enum CheckResult {
    EQUAL = 0,
    SANE = 1,
    INSANE = 2,
    REFRESHED = 3,
  };

  Database* const db_;
  LogVerifier* const verifier_;
  cert_trans::HTTPLogClient* const client_;
  const uint64_t sleep_time_;

  VerifyResult VerifySTHInternal();
  VerifyResult VerifySTHInternal(const ct::SignedTreeHead& sth);

  ConfirmResult ConfirmTreeInternal();
  ConfirmResult ConfirmTreeInternal(const ct::SignedTreeHead& sth);

  // Checks if two (subsequent) STHs are sane regarding timestamp and tree
  // size.
  // Prerequisite: Both STHs should have a valid signature and not be
  // malformed.
  // Only used internaly in loop().
  CheckResult CheckSTHSanity(const ct::SignedTreeHead& old_sth,
                             const ct::SignedTreeHead& new_sth);

  VerifyResult VerifySTHWithInvalidTimestamp(const ct::SignedTreeHead& sth);

  DISALLOW_COPY_AND_ASSIGN(Monitor);
};

}  // namespace monitor

#endif  // MONITOR_H
