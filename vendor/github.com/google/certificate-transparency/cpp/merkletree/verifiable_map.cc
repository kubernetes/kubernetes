#include <array>
#include <string>

#include "merkletree/verifiable_map.h"


using std::string;
using std::unique_ptr;
using std::vector;
using util::Status;
using util::StatusOr;


namespace cert_trans {


VerifiableMap::VerifiableMap(SerialHasher* hasher)
    : hasher_model_(CHECK_NOTNULL(hasher)->Create()), merkle_tree_(hasher) {
}


void VerifiableMap::Set(const string& key, const string& value) {
  const SparseMerkleTree::Path path(PathFromKey(key));
  merkle_tree_.SetLeaf(path, value);
  values_[path] = value;
}


StatusOr<string> VerifiableMap::Get(const string& key) const {
  const SparseMerkleTree::Path path(PathFromKey(key));
  const auto it(values_.find(path));
  if (it == values_.end()) {
    return Status(util::error::NOT_FOUND, "No such entry.");
  }
  return it->second;
}


vector<string> VerifiableMap::InclusionProof(const string& key) {
  return merkle_tree_.InclusionProof(PathFromKey(key));
}


SparseMerkleTree::Path VerifiableMap::PathFromKey(const string& key) const {
  unique_ptr<SerialHasher> h(hasher_model_->Create());
  h->Update(key);
  return PathFromBytes(h->Final());
}

}  // namespace cert_trans
