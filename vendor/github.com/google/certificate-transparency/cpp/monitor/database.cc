#include "monitor/database.h"

#include "merkletree/tree_hasher.h"
#include "proto/serializer.h"

namespace monitor {

Database::WriteResult Database::CreateEntry(
    const cert_trans::LoggedEntry& logged) {
  std::string leaf;
  if (!logged.SerializeForLeaf(&leaf))
    return this->SERIALIZE_FAILED;

  TreeHasher hasher(new Sha256Hasher);
  std::string leaf_hash = hasher.HashLeaf(leaf);

  std::string cert = Serializer::LeafData(logged.entry());

  std::string cert_chain;
  if (!logged.SerializeExtraData(&cert_chain))
    return this->SERIALIZE_FAILED;

  return CreateEntry_(leaf, leaf_hash, cert, cert_chain);
}

Database::WriteResult Database::WriteSTH(const ct::SignedTreeHead& sth) {
  CHECK(sth.has_timestamp());
  CHECK(sth.has_tree_size());

  // Serialzing is not TLS (RFC) conform.
  return WriteSTH_(sth.timestamp(), sth.tree_size(), sth.SerializeAsString());
}

Database::WriteResult Database::SetVerificationLevel(
    const ct::SignedTreeHead& sth, VerificationLevel verify_level) {
  if (verify_level == this->UNDEFINED)
    return this->NOT_ALLOWED;

  return SetVerificationLevel_(sth, verify_level);
}

}  // namespace monitor
