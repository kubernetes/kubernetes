/* -*- indent-tabs-mode: nil -*- */
#include "log/frontend.h"

#include <glog/logging.h>

#include "log/cert.h"
#include "log/cert_submission_handler.h"
#include "log/frontend_signer.h"
#include "monitoring/event_metric.h"
#include "proto/ct.pb.h"
#include "util/status.h"

using cert_trans::CertChain;
using cert_trans::PreCertChain;
using ct::LogEntry;
using ct::SignedCertificateTimestamp;
using std::string;
using std::lock_guard;
using std::mutex;
using util::Status;

namespace {

static cert_trans::EventMetric<std::string, std::string>
    submission_status_metric(
        "submission_status", "entry_type", "status",
        "Submission status totals broken down by entry type and status code.");

Status UpdateStats(ct::LogEntryType type, const Status& status) {
  if (type == ct::X509_ENTRY) {
    submission_status_metric.RecordEvent(
        "x509", util::ErrorCodeString(status.CanonicalCode()), 1);
  } else {
    submission_status_metric.RecordEvent(
        "precert", util::ErrorCodeString(status.CanonicalCode()), 1);
  }
  return status;
}

}  // namespace

Frontend::Frontend(FrontendSigner* signer) : signer_(CHECK_NOTNULL(signer)) {
}

Frontend::~Frontend() {
}

Status Frontend::QueueProcessedEntry(Status pre_status, const LogEntry& entry,
                                     SignedCertificateTimestamp* sct) {
  CHECK(entry.has_type());
  if (!pre_status.ok()) {
    return UpdateStats(entry.type(), pre_status);
  }

  // Step 2. Submit to database.
  return UpdateStats(entry.type(), signer_->QueueEntry(entry, sct));
}
