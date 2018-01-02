#include "monitor/monitor.h"

#include "client/http_log_client.h"
#include "log/log_verifier.h"
#include "merkletree/merkle_tree.h"
#include "monitor/database.h"

using cert_trans::AsyncLogClient;
using cert_trans::HTTPLogClient;
using std::string;

namespace monitor {

Monitor::Monitor(Database* database, LogVerifier* log_verifier,
                 HTTPLogClient* client, uint64_t sleep_time_sec)
    : db_(CHECK_NOTNULL(database)),
      verifier_(CHECK_NOTNULL(log_verifier)),
      client_(CHECK_NOTNULL(client)),
      sleep_time_(sleep_time_sec) {
}

Monitor::GetResult Monitor::GetSTH() {
  ct::SignedTreeHead new_sth;

  if (client_->GetSTH(&new_sth) != AsyncLogClient::OK)
    return NETWORK_PROBLEM;

  ct::SignedTreeHead current_sth;
  Database::LookupResult ret;
  ret = db_->LookupLatestWrittenSTH(&current_sth);

  // ct::SignedTreeHead::SerializeAsString() returns an empty string on
  // failure.
  // This might lead to unexpected behaviour (i.e. a database write).
  if (ret == Database::NOT_FOUND ||
      current_sth.SerializeAsString() != new_sth.SerializeAsString()) {
    CHECK_EQ(db_->WriteSTH(new_sth), Database::WRITE_OK);

    if (ret == Database::NOT_FOUND) {
      LOG(INFO) << "NEW DATABASE!";
    } else {
      CHECK_EQ(ret, Database::LOOKUP_OK);

      LOG(INFO) << "current STH:";
      LOG(INFO) << current_sth.timestamp();
      LOG(INFO) << current_sth.tree_size();
      LOG(INFO) << util::ToBase64(current_sth.sha256_root_hash());
    }
    LOG(INFO) << "new STH:";
    LOG(INFO) << new_sth.timestamp();
    LOG(INFO) << new_sth.tree_size();
    LOG(INFO) << util::ToBase64(new_sth.sha256_root_hash());
  } else {
    CHECK_EQ(ret, Database::LOOKUP_OK);
    LOG(INFO) << "STH unchanged";
  }
  return OK;
}

Monitor::VerifyResult Monitor::VerifySTH(uint64_t timestamp) {
  ct::SignedTreeHead sth;

  if (timestamp) {
    CHECK_EQ(db_->LookupSTHByTimestamp(timestamp, &sth), Database::LOOKUP_OK);
  } else {
    const GetResult result(GetSTH());
    switch (result) {
      case OK:
        break;
      case NETWORK_PROBLEM:
        LOG(WARNING) << "network problem";
        break;
      default:
        LOG(FATAL) << "unknown result from Monitor::GetSTH: " << result;
    }
    CHECK_EQ(db_->LookupLatestWrittenSTH(&sth), Database::LOOKUP_OK);
  }
  return VerifySTHInternal(sth);
}

Monitor::VerifyResult Monitor::VerifySTHInternal() {
  return VerifySTH(0);
}

Monitor::VerifyResult Monitor::VerifySTHWithInvalidTimestamp(
    const ct::SignedTreeHead& sth) {
  LogVerifier::LogVerifyResult v_result =
      verifier_->VerifySignedTreeHead(sth, sth.timestamp(), sth.timestamp());

  if (v_result == LogVerifier::VERIFY_OK) {
    return STH_MALFORMED_WTH_VALID_SIGNATURE;
  } else if (v_result == LogVerifier::INVALID_SIGNATURE) {
    // If the signature is invalid we don't care about the timestamp
    // (i.e. STH might not be from the log).
    return SIGNATURE_INVALID;
  } else {
    LOG(FATAL) << "Unknown verification error: "
               << LogVerifier::VerifyResultString(v_result);
  }
}

Monitor::VerifyResult Monitor::VerifySTHInternal(
    const ct::SignedTreeHead& sth) {
  LogVerifier::LogVerifyResult v_result = verifier_->VerifySignedTreeHead(sth);
  std::string v_result_string = LogVerifier::VerifyResultString(v_result);

  VerifyResult result = SIGNATURE_INVALID;
  Database::VerificationLevel level = Database::SIGNATURE_VERIFICATION_FAILED;

  if (v_result == LogVerifier::VERIFY_OK) {
    result = SIGNATURE_VALID;
    level = Database::SIGNATURE_VERIFIED;
  } else if (v_result == LogVerifier::INVALID_SIGNATURE) {
    result = SIGNATURE_INVALID;
    level = Database::SIGNATURE_VERIFICATION_FAILED;
  } else if (v_result == LogVerifier::INVALID_TIMESTAMP) {
    LOG(INFO) << "verify   : " << v_result_string;

    // In case of an invalid timestamp, nevertheless verify the signature
    // because we want to know if this (broken) STH is signed by the log.
    result = VerifySTHWithInvalidTimestamp(sth);
    if (result == SIGNATURE_VALID)
      level = Database::INCONSISTENT;
  } else {
    LOG(FATAL) << "Unknown verification error: " << v_result_string;
  }

  LOG(INFO) << "verify   : " << v_result_string;
  LOG(INFO) << "tree size: " << sth.tree_size();
  LOG(INFO) << "timestamp: " << sth.timestamp();
  LOG(INFO) << "root hash: " << util::ToBase64(sth.sha256_root_hash());

  Database::VerificationLevel current_level;
  CHECK_EQ(db_->LookupVerificationLevel(sth, &current_level),
           Database::LOOKUP_OK);
  LOG(INFO) << "Previous verification level: "
            << Database::VerificationLevelString(current_level);

  // Only overwrite the verification level in the database if the STH is not
  // yet verified or the verification failed previously.
  if (current_level == Database::UNDEFINED ||
      current_level == Database::SIGNATURE_VERIFICATION_FAILED) {
    CHECK_EQ(db_->SetVerificationLevel(sth, level), Database::WRITE_OK);
    LOG(INFO) << "New verification level: "
              << Database::VerificationLevelString(level);
  }
  return result;
}

Monitor::GetResult Monitor::GetEntries(int get_first, int get_last) {
  CHECK(get_first >= 0);
  CHECK(get_last >= get_first);

  std::vector<AsyncLogClient::Entry> entries;
  int dload_count = 0;
  do {
    // If the server does not impose a limit, all entries from get_first to
    // get_last will be downloaded at once (could exceed memory).
    AsyncLogClient::Status error =
        client_->GetEntries(get_first + dload_count, get_last, &entries);

    if (error != AsyncLogClient::OK) {
      LOG(ERROR) << "HTTPLogClient returned with error " << error
                 << ". No entries have been written to the database.";
      return NETWORK_PROBLEM;
    }

    LOG(INFO) << "Writing entries from " << get_first + dload_count << " to "
              << get_first + dload_count + entries.size();
    dload_count += entries.size();

    db_->BeginTransaction();
    for (size_t i = 0; i < entries.size(); i++) {
      cert_trans::LoggedEntry logged;
      CHECK(logged.CopyFromClientLogEntry(entries.at(i)));
      CHECK_EQ(db_->CreateEntry(logged), Database::WRITE_OK);
    }
    entries.clear();
    db_->EndTransaction();

  } while (dload_count + get_first <= get_last);
  return OK;
}

Monitor::ConfirmResult Monitor::ConfirmTree(uint64_t timestamp) {
  ct::SignedTreeHead sth;

  if (timestamp) {
    CHECK_EQ(db_->LookupSTHByTimestamp(timestamp, &sth), Database::LOOKUP_OK);
  } else {
    CHECK_EQ(db_->LookupLatestWrittenSTH(&sth), Database::LOOKUP_OK);
  }
  return ConfirmTreeInternal(sth);
}

Monitor::ConfirmResult Monitor::ConfirmTreeInternal() {
  return ConfirmTree(0);
}

Monitor::ConfirmResult Monitor::ConfirmTreeInternal(
    const ct::SignedTreeHead& sth) {
  MerkleTree mt(new Sha256Hasher);

  Database::VerificationLevel lvl;
  CHECK_EQ(db_->LookupVerificationLevel(sth, &lvl), Database::LOOKUP_OK);
  CHECK_EQ(lvl, Database::SIGNATURE_VERIFIED);

  std::string hash;

  LOG(INFO) << "Building tree...";

  for (int64_t current = 1; current <= sth.tree_size(); current++) {
    CHECK_EQ(db_->LookupHashByIndex(current, &hash), Database::LOOKUP_OK);
    mt.AddLeafHash(hash);
  }

  LOG(INFO) << "merkle tree_size and root_hash:";
  LOG(INFO) << mt.LeafCount();
  LOG(INFO) << util::ToBase64(mt.CurrentRoot());
  LOG(INFO) << "STH tree_size and root_hash:";
  LOG(INFO) << sth.tree_size();
  LOG(INFO) << util::ToBase64(sth.sha256_root_hash());

  if (mt.CurrentRoot() != sth.sha256_root_hash()) {
    LOG(ERROR) << "Tree confirmation failed - hashes mismatch.";
    CHECK_EQ(db_->SetVerificationLevel(sth,
                                       Database::TREE_CONFIRMATION_FAILED),
             Database::WRITE_OK);

    return TREE_CONFIRMATION_FAILED;
  }

  CHECK_EQ(db_->SetVerificationLevel(sth, Database::TREE_CONFIRMED),
           Database::WRITE_OK);
  LOG(INFO) << "Tree confirmed.";
  return TREE_CONFIRMED;
}

Monitor::CheckResult Monitor::CheckSTHSanity(
    const ct::SignedTreeHead& old_sth, const ct::SignedTreeHead& new_sth) {
  // This serializing returns an empty String on failure which will lead to
  // undefined behaviour.
  if (old_sth.SerializeAsString() == new_sth.SerializeAsString())
    return EQUAL;

  LOG(INFO) << "Retrieved new STH...";

  if (new_sth.timestamp() <= old_sth.timestamp() ||
      new_sth.tree_size() < old_sth.tree_size()) {
    if (new_sth.timestamp() <= old_sth.timestamp())
      LOG(ERROR) << "New timestamp not newer than old one.";
    if (new_sth.tree_size() < old_sth.tree_size())
      LOG(ERROR) << "New tree size smaller than old one.";

    CHECK_EQ(db_->SetVerificationLevel(new_sth, Database::INCONSISTENT),
             Database::WRITE_OK);
    return INSANE;
  }

  if (new_sth.timestamp() > old_sth.timestamp() &&
      new_sth.tree_size() == old_sth.tree_size() &&
      new_sth.sha256_root_hash().compare(old_sth.sha256_root_hash()) == 0) {
    LOG(INFO) << " just with a fresh timestamp.";
    return REFRESHED;
  }

  LOG(INFO) << " sanity checks passed.";
  return SANE;
}

void Monitor::Init() {
  CHECK_EQ(VerifySTHInternal(), SIGNATURE_VALID);

  ct::SignedTreeHead sth;
  CHECK_EQ(db_->LookupLatestWrittenSTH(&sth), Database::LOOKUP_OK);

  CHECK_EQ(GetEntries(0, sth.tree_size() - 1), OK);
  ConfirmTreeInternal();
}

void Monitor::Loop() {
  ct::SignedTreeHead old_sth;
  ct::SignedTreeHead new_sth;

  if (db_->LookupLatestWrittenSTH(&old_sth), Database::LOOKUP_OK)
    LOG(FATAL) << "Run init_monitor first.";

  while (true) {
    LOG(INFO) << "Sleeping for " << sleep_time_ << " seconds.";
    // TODO(weidner): Better only sleep sleep_time - time_used_in_loop.
    sleep(sleep_time_);

    if (VerifySTHInternal() != SIGNATURE_VALID)
      continue;

    CHECK_EQ(db_->LookupLatestWrittenSTH(&new_sth), Database::LOOKUP_OK);

    const CheckResult sanity(CheckSTHSanity(old_sth, new_sth));
    if (sanity == SANE) {
      if (GetEntries(old_sth.tree_size(), new_sth.tree_size() - 1) != OK) {
        continue;
      }
    }
    if (sanity == REFRESHED || sanity == SANE) {
      // Go on even the confirmation fails to continue to monitor the log.
      // Nevertheless the failure is logged and written to the database.
      ConfirmTreeInternal();

      old_sth = new_sth;
    }
  }
}

}  // namespace monitor
