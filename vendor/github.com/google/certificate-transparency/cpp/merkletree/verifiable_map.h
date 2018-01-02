#ifndef CERT_TRANS_MERKLETREE_VERIFIABLE_MAP_H_
#define CERT_TRANS_MERKLETREE_VERIFIABLE_MAP_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "base/macros.h"
#include "merkletree/sparse_merkle_tree.h"
#include "util/statusor.h"

namespace cert_trans {


// Implements a Verifiable Map using a SparseMerkleTree and hashmap.
class VerifiableMap {
 public:
  VerifiableMap(SerialHasher* hasher);

  std::string CurrentRoot() {
    return merkle_tree_.CurrentRoot();
  }

  void Set(const std::string& key, const std::string& value);

  util::StatusOr<std::string> Get(const std::string& key) const;

  std::vector<std::string> InclusionProof(const std::string& key);

 private:
  SparseMerkleTree::Path PathFromKey(const std::string& key) const;

  std::unique_ptr<SerialHasher> hasher_model_;
  SparseMerkleTree merkle_tree_;

  // TODO(alcutter): allow arbitrary stores here.
  std::unordered_map<SparseMerkleTree::Path, std::string, PathHasher> values_;

  DISALLOW_COPY_AND_ASSIGN(VerifiableMap);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_MERKLETREE_VERIFIABLE_MAP_H_
