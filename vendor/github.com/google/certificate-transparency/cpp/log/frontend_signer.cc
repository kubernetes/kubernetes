/* -*- indent-tabs-mode: nil -*- */
#include "log/frontend_signer.h"

#include <glog/logging.h>

#include "log/database.h"
#include "log/log_signer.h"
#include "merkletree/serial_hasher.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/status.h"
#include "util/util.h"

using cert_trans::ConsistentStore;
using cert_trans::Database;
using cert_trans::LoggedEntry;
using ct::LogEntry;
using ct::SignedCertificateTimestamp;
using std::string;
using util::Status;


FrontendSigner::FrontendSigner(Database* db,
                               ConsistentStore<LoggedEntry>* store,
                               LogSigner* signer)
    : db_(CHECK_NOTNULL(db)),
      store_(CHECK_NOTNULL(store)),
      signer_(CHECK_NOTNULL(signer)) {
}

Status FrontendSigner::QueueEntry(const LogEntry& entry,
                                  SignedCertificateTimestamp* sct) {
  const string sha256_hash(
      Sha256Hasher::Sha256Digest(Serializer::LeafData(entry)));
  CHECK(!sha256_hash.empty());

  // Check if the entry already exists in the local DB (i.e. it's been
  // integrated into the tree.)
  // This isn't foolproof; it could be that the local node doesn't yet have
  // a copy of this if the cert was added recently, but it's not fatal if the
  // same cert gets added twice.
  // TODO(ekasper): switch to using SignedEntryWithType as the DB key.
  cert_trans::LoggedEntry logged;
  Database::LookupResult db_result = db_->LookupByHash(sha256_hash, &logged);

  if (db_result == Database::LOOKUP_OK) {
    // If we did find a local copy, return the previously issued SCT.
    if (sct != nullptr) {
      *sct = logged.sct();
    }
    return Status(util::error::ALREADY_EXISTS,
                  "entry already exists in Database");
  }
  CHECK_EQ(Database::NOT_FOUND, db_result);

  // Dont have the cert locally, so create an SCT and store it and the cert.
  SignedCertificateTimestamp local_sct;
  TimestampAndSign(entry, &local_sct);

  cert_trans::LoggedEntry new_logged;
  new_logged.mutable_sct()->CopyFrom(local_sct);
  new_logged.mutable_entry()->CopyFrom(entry);
  CHECK_EQ(new_logged.Hash(), sha256_hash);

  // If this cert has already been added (but not yet integrated into the
  // tree), then this call will update new_logged.sct with the previously
  // issued one.
  util::Status status(store_->AddPendingEntry(&new_logged));
  CHECK_EQ(new_logged.Hash(), sha256_hash);

  if (sct != nullptr) {
    *sct = new_logged.sct();
  }

  return status;
}


void FrontendSigner::TimestampAndSign(const LogEntry& entry,
                                      SignedCertificateTimestamp* sct) const {
  sct->set_version(ct::V1);
  sct->set_timestamp(util::TimeInMilliseconds());
  sct->clear_extensions();
  // The submission handler has already verified the format of this entry,
  // so this should never fail.
  CHECK_EQ(LogSigner::OK, signer_->SignCertificateTimestamp(entry, sct));
}
