#ifndef CERT_SUBMISSION_HANDLER_H
#define CERT_SUBMISSION_HANDLER_H

#include <string>

#include "base/macros.h"
#include "log/cert_checker.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/status.h"

namespace cert_trans {


// Parse incoming submissions, do preliminary sanity checks and pass them
// through cert checker.
// Prepare for signing by parsing the input into an appropriate
// log entry structure.
class CertSubmissionHandler {
 public:
  // Does not take ownership of the cert_checker.
  explicit CertSubmissionHandler(const cert_trans::CertChecker* cert_checker);

  // These may change |chain|.
  // TODO(pphaneuf): These could return StatusOr<ct::LogEntry>.
  util::Status ProcessX509Submission(cert_trans::CertChain* chain,
                                     ct::LogEntry* entry) const;
  util::Status ProcessPreCertSubmission(cert_trans::PreCertChain* chain,
                                        ct::LogEntry* entry) const;

  // For clients, to reconstruct the bytestring under the signature
  // from the observed chain. Does not check whether the entry
  // has valid format (i.e., does not check length limits).
  static bool X509ChainToEntry(const cert_trans::CertChain& chain,
                               ct::LogEntry* entry);

 private:
  const cert_trans::CertChecker* const cert_checker_;

  DISALLOW_COPY_AND_ASSIGN(CertSubmissionHandler);
};


}  // namespace cert_trans

#endif
