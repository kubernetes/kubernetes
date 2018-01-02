#include "log/cert_submission_handler.h"

#include <glog/logging.h>
#include <string>

#include "log/cert.h"
#include "log/cert_checker.h"
#include "log/ct_extensions.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"

using cert_trans::Cert;
using cert_trans::CertChain;
using cert_trans::CertChecker;
using cert_trans::PreCertChain;
using cert_trans::TbsCertificate;
using ct::LogEntry;
using ct::PrecertChainEntry;
using ct::X509ChainEntry;
using std::string;
using util::Status;
using util::StatusOr;

namespace cert_trans {
namespace {


bool SerializedTbs(const Cert& cert, string* result) {
  if (!cert.IsLoaded()) {
    return false;
  }

  const StatusOr<bool> has_embedded_proof = cert.HasExtension(
      cert_trans::NID_ctEmbeddedSignedCertificateTimestampList);
  if (!has_embedded_proof.ok()) {
    return false;
  }

  // Delete the embedded proof.
  TbsCertificate tbs(cert);
  if (!tbs.IsLoaded()) {
    return false;
  }

  if (has_embedded_proof.ValueOrDie() &&
      !tbs.DeleteExtension(
              cert_trans::NID_ctEmbeddedSignedCertificateTimestampList)
           .ok()) {
    return false;
  }

  string der_tbs;
  if (!tbs.DerEncoding(&der_tbs).ok()) {
    return false;
  }

  result->assign(der_tbs);
  return true;
}


}  // namespace


// TODO(ekasper): handle Cert errors consistently and log some errors here
// if they fail.
CertSubmissionHandler::CertSubmissionHandler(const CertChecker* cert_checker)
    : cert_checker_(CHECK_NOTNULL(cert_checker)) {
}


// static
bool CertSubmissionHandler::X509ChainToEntry(const CertChain& chain,
                                             LogEntry* entry) {
  if (!chain.IsLoaded()) {
    return false;
  }

  const StatusOr<bool> has_embedded_proof = chain.LeafCert()->HasExtension(
      cert_trans::NID_ctEmbeddedSignedCertificateTimestampList);
  if (!has_embedded_proof.ok()) {
    LOG(ERROR) << "Failed to check embedded SCT extension.";
    return false;
  }

  if (has_embedded_proof.ValueOrDie()) {
    if (chain.Length() < 2) {
      // need issuer
      return false;
    }

    entry->set_type(ct::PRECERT_ENTRY);
    string key_hash;
    if (chain.CertAt(1)->SPKISha256Digest(&key_hash) != Status::OK) {
      return false;
    }

    entry->mutable_precert_entry()->mutable_pre_cert()->set_issuer_key_hash(
        key_hash);

    string tbs;
    if (!SerializedTbs(*chain.LeafCert(), &tbs))
      return false;

    entry->mutable_precert_entry()->mutable_pre_cert()->set_tbs_certificate(
        tbs);
    return true;
  } else {
    entry->set_type(ct::X509_ENTRY);
    string der_cert;
    if (chain.LeafCert()->DerEncoding(&der_cert) != Status::OK) {
      return false;
    }

    entry->mutable_x509_entry()->set_leaf_certificate(der_cert);
    return true;
  }
}


Status CertSubmissionHandler::ProcessX509Submission(CertChain* chain,
                                                    LogEntry* entry) const {
  entry->set_type(ct::X509_ENTRY);
  if (!chain->IsLoaded())
    return Status(util::error::INVALID_ARGUMENT, "empty submission");

  const Status status(cert_checker_->CheckCertChain(chain));
  if (!status.ok())
    return status;

  // We have a valid chain; make the entry.
  string der_cert;
  // Nothing should fail anymore as we have validated the chain.
  if (chain->LeafCert()->DerEncoding(&der_cert) != Status::OK) {
    return Status(util::error::INTERNAL, "could not DER-encode the chain");
  }

  X509ChainEntry* x509_entry = entry->mutable_x509_entry();
  x509_entry->set_leaf_certificate(der_cert);
  for (size_t i = 1; i < chain->Length(); ++i) {
    if (chain->CertAt(i)->DerEncoding(&der_cert) != Status::OK) {
      return Status(util::error::INTERNAL, "could not DER-encode the chain");
    }
    x509_entry->add_certificate_chain(der_cert);
  }
  return Status::OK;
}


Status CertSubmissionHandler::ProcessPreCertSubmission(PreCertChain* chain,
                                                       LogEntry* entry) const {
  entry->set_type(ct::PRECERT_ENTRY);
  PrecertChainEntry* precert_entry = entry->mutable_precert_entry();
  const Status status(cert_checker_->CheckPreCertChain(
      chain, precert_entry->mutable_pre_cert()->mutable_issuer_key_hash(),
      precert_entry->mutable_pre_cert()->mutable_tbs_certificate()));

  if (!status.ok())
    return status;

  // We have a valid chain; make the entry.
  string der_cert;
  // Nothing should fail anymore as we have validated the chain.
  if (chain->LeafCert()->DerEncoding(&der_cert) != Status::OK) {
    return Status(util::error::INTERNAL, "could not DER-encode the chain");
  }
  precert_entry->set_pre_certificate(der_cert);
  for (size_t i = 1; i < chain->Length(); ++i) {
    if (chain->CertAt(i)->DerEncoding(&der_cert) != Status::OK)
      return Status(util::error::INTERNAL, "could not DER-encode the chain");
    precert_entry->add_precertificate_chain(der_cert);
  }
  return Status::OK;
}


}  // namespace cert_trans
