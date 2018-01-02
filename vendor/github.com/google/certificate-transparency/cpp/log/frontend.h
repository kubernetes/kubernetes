/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef FRONTEND_H
#define FRONTEND_H

#include <memory>
#include <mutex>

#include "base/macros.h"
#include "log/cert.h"
#include "log/submit_result.h"
#include "proto/ct.pb.h"

class FrontendSigner;

namespace util {
class Status;
}  // namespace util

// Frontend for accepting new submissions.
class Frontend {
 public:
  // Takes ownership of the signer.
  Frontend(FrontendSigner* signer);
  ~Frontend();

  util::Status QueueProcessedEntry(util::Status pre_status,
                                   const ct::LogEntry& entry,
                                   ct::SignedCertificateTimestamp* sct);

 private:
  const std::unique_ptr<FrontendSigner> signer_;

  DISALLOW_COPY_AND_ASSIGN(Frontend);
};
#endif
