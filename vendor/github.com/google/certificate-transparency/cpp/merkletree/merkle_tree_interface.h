// An interface for Merkle trees.  It makes it easier to write code that works
// with all all flavors of Merkle trees.

#ifndef SRC_MERKLETREE_MERKLE_TREE_INTERFACE_H_
#define SRC_MERKLETREE_MERKLE_TREE_INTERFACE_H_

#include <stddef.h>
#include <string>

#include "base/macros.h"

namespace cert_trans {

// An interface for Merkle trees.  See specializations in
// merkletree/merkle_tree.h and merkletree/compact_merkle_tree.h.
class MerkleTreeInterface {
 public:
  MerkleTreeInterface() = default;
  virtual ~MerkleTreeInterface() = default;

  // Length of a node (i.e., a hash), in bytes.
  virtual size_t NodeSize() const = 0;

  // Number of leaves in the tree.
  virtual size_t LeafCount() const = 0;

  // Returns the leaf hash, but do not append the data to the tree.
  virtual std::string LeafHash(const std::string& data) const = 0;

  // Number of levels. An empty tree has 0 levels, a tree with 1 leaf has
  // 1 level, a tree with 2 leaves has 2 levels, and a tree with n leaves has
  // ceil(log2(n)) + 1 levels.
  virtual size_t LevelCount() const = 0;

  // Add a new leaf to the hash tree.
  //
  // Returns the position of the leaf in the tree. Indexing starts at 1,
  // so position = number of leaves in the tree after this update.
  //
  // @param data Binary input blob
  virtual size_t AddLeaf(const std::string& data) = 0;

  // Add a new leaf to the hash tree. It is the caller's responsibility
  // to ensure that the hash is correct.
  //
  // Returns the position of the leaf in the tree. Indexing starts at 1,
  // so position = number of leaves in the tree after this update.
  //
  // @param hash leaf hash
  virtual size_t AddLeafHash(const std::string& hash) = 0;

  // Get the current root of the tree.
  // Update the root to reflect the current shape of the tree,
  // and return the tree digest.
  //
  // Returns the hash of an empty string if the tree has no leaves
  // (and hence, no root).
  virtual std::string CurrentRoot() = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(MerkleTreeInterface);
};

}  // namespace cert_trans

#endif  // SRC_MERKLETREE_MERKLE_TREE_INTERFACE_H_
