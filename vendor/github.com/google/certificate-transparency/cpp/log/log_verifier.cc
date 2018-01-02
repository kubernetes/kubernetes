#include "log/log_verifier.h"

#include <glog/logging.h>
#include <stdint.h>

#include "log/cert_submission_handler.h"
#include "log/log_signer.h"
#include "merkletree/merkle_verifier.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/util.h"

using ct::LogEntry;
using ct::MerkleAuditProof;
using ct::SignedCertificateTimestamp;
using ct::SignedTreeHead;
using std::string;

LogVerifier::LogVerifier(LogSigVerifier* sig_verifier,
                         MerkleVerifier* merkle_verifier)
    : sig_verifier_(sig_verifier), merkle_verifier_(merkle_verifier) {
}

LogVerifier::~LogVerifier() {
  delete sig_verifier_;
  delete merkle_verifier_;
}

LogVerifier::LogVerifyResult LogVerifier::VerifySignedCertificateTimestamp(
    const LogEntry& entry, const SignedCertificateTimestamp& sct,
    uint64_t begin_range, uint64_t end_range, string* merkle_leaf_hash) const {
  if (!IsBetween(sct.timestamp(), begin_range, end_range))
    return INVALID_TIMESTAMP;

  // TODO(ekasper): separate format and signature errors.
  if (sig_verifier_->VerifySCTSignature(entry, sct) != LogSigVerifier::OK)
    return INVALID_SIGNATURE;
  string serialized_leaf;
  // If SCT verification succeeded, then we should never fail here.
  if (merkle_leaf_hash != NULL) {
    CHECK_EQ(SerializeResult::OK,
             Serializer::SerializeSCTMerkleTreeLeaf(sct, entry,
                                                    &serialized_leaf));
    merkle_leaf_hash->assign(merkle_verifier_->LeafHash(serialized_leaf));
  }
  return VERIFY_OK;
}

LogVerifier::LogVerifyResult LogVerifier::VerifySignedCertificateTimestamp(
    const LogEntry& entry, const SignedCertificateTimestamp& sct,
    string* merkle_leaf_hash) const {
  // Allow a bit of slack, say 1 second into the future.
  return VerifySignedCertificateTimestamp(entry, sct, 0,
                                          util::TimeInMilliseconds() + 1000,
                                          merkle_leaf_hash);
}

LogVerifier::LogVerifyResult LogVerifier::VerifySignedTreeHead(
    const SignedTreeHead& sth, uint64_t begin_range,
    uint64_t end_range) const {
  if (!IsBetween(sth.timestamp(), begin_range, end_range))
    return INVALID_TIMESTAMP;

  if (sig_verifier_->VerifySTHSignature(sth) != LogSigVerifier::OK)
    return INVALID_SIGNATURE;
  return VERIFY_OK;
}

LogVerifier::LogVerifyResult LogVerifier::VerifySignedTreeHead(
    const SignedTreeHead& sth) const {
  // Allow a bit of slack, say 1 second into the future.
  return VerifySignedTreeHead(sth, 0, util::TimeInMilliseconds() + 1000);
}

LogVerifier::LogVerifyResult LogVerifier::VerifyMerkleAuditProof(
    const LogEntry& entry, const SignedCertificateTimestamp& sct,
    const MerkleAuditProof& merkle_proof) const {
  if (!IsBetween(merkle_proof.timestamp(), sct.timestamp(),
                 util::TimeInMilliseconds() + 1000))
    return INCONSISTENT_TIMESTAMPS;

  string serialized_leaf;
  SerializeResult serialize_result =
      Serializer::SerializeSCTMerkleTreeLeaf(sct, entry, &serialized_leaf);

  if (serialize_result != SerializeResult::OK)
    return INVALID_FORMAT;

  std::vector<string> path;
  for (int i = 0; i < merkle_proof.path_node_size(); ++i)
    path.push_back(merkle_proof.path_node(i));

  // Leaf indexing in the MerkleTree starts from 1.
  string root_hash =
      merkle_verifier_->RootFromPath(merkle_proof.leaf_index() + 1,
                                     merkle_proof.tree_size(), path,
                                     serialized_leaf);

  if (root_hash.empty())
    return INVALID_MERKLE_PATH;

  SignedTreeHead sth;
  sth.set_version(merkle_proof.version());
  sth.mutable_id()->CopyFrom(merkle_proof.id());
  sth.set_timestamp(merkle_proof.timestamp());
  sth.set_tree_size(merkle_proof.tree_size());
  sth.set_sha256_root_hash(root_hash);
  sth.mutable_signature()->CopyFrom(merkle_proof.tree_head_signature());

  if (sig_verifier_->VerifySTHSignature(sth) != LogSigVerifier::OK)
    return INVALID_SIGNATURE;
  return VERIFY_OK;
}

/* static */
bool LogVerifier::IsBetween(uint64_t timestamp, uint64_t earliest,
                            uint64_t latest) {
  return timestamp >= earliest && timestamp <= latest;
}

bool LogVerifier::VerifyConsistency(
    const ct::SignedTreeHead& sth1, const ct::SignedTreeHead& sth2,
    const std::vector<std::string>& proof) const {
  return merkle_verifier_->VerifyConsistency(sth1.tree_size(),
                                             sth2.tree_size(),
                                             sth1.sha256_root_hash(),
                                             sth2.sha256_root_hash(), proof);
}
