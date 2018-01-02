#ifndef COMPACT_MERKLETREE_H
#define COMPACT_MERKLETREE_H

#include <stddef.h>
#include <string>
#include <vector>

#include "merkletree/merkle_tree.h"
#include "merkletree/merkle_tree_interface.h"
#include "merkletree/tree_hasher.h"

class SerialHasher;

// A memory-efficient version of Merkle Trees; like MerkleTree
// (see merkletree/merkle_tree.h) but can only add new leaves and report
// its current root (i.e., it cannot do paths, snapshots or consistency).
//
// This class is thread-compatible, but not thread-safe.
class CompactMerkleTree : public cert_trans::MerkleTreeInterface {
 public:
  // The constructor takes a pointer to some concrete hash function
  // instantiation of the SerialHasher abstract class.
  // Takes ownership of the hasher.
  explicit CompactMerkleTree(SerialHasher* hasher);
  CompactMerkleTree(const CompactMerkleTree& other, SerialHasher* hasher);

  explicit CompactMerkleTree(CompactMerkleTree&& other) = default;

  // Creates a new CompactMerkleTree based on the data present in the
  // (non-compact) MerkleTree |model|.
  // Takes ownership of |hasher|.
  CompactMerkleTree(MerkleTree& model, SerialHasher* hasher);

  virtual ~CompactMerkleTree();

  // Length of a node (i.e., a hash), in bytes.
  virtual size_t NodeSize() const {
    return treehasher_.DigestSize();
  };

  // Number of leaves in the tree.
  virtual size_t LeafCount() const {
    return leaf_count_;
  }

  // Return the leaf hash, but do not append the data to the tree.
  virtual std::string LeafHash(const std::string& data) const {
    return treehasher_.HashLeaf(data);
  }

  // Number of levels. An empty tree has 0 levels, a tree with 1 leaf has
  // 1 level, a tree with 2 leaves has 2 levels, and a tree with n leaves has
  // ceil(log2(n)) + 1 levels.
  virtual size_t LevelCount() const {
    return level_count_;
  }

  // Add a new leaf to the hash tree.
  //
  // (We update intermediate hashes as soon as a node becomes "fixed"
  // (However, we evaluate the root lazily, and do not update it here.)
  //
  // Returns the position of the leaf in the tree. Indexing starts at 1,
  // so position = number of leaves in the tree after this update.
  //
  // @param data Binary input blob
  virtual size_t AddLeaf(const std::string& data);

  // Add a new leaf to the hash tree. It is the caller's responsibility
  // to ensure that the hash is correct.
  //
  // (We update intermediate hashes as soon as a node becomes "fixed"
  // (However, we evaluate the root lazily, and do not update it here.)
  //
  // Returns the position of the leaf in the tree. Indexing starts at 1,
  // so position = number of leaves in the tree after this update.
  //
  // @param hash leaf hash
  virtual size_t AddLeafHash(const std::string& hash);

  // Get the current root of the tree.
  // Update the root to reflect the current shape of the tree,
  // and return the tree digest.
  //
  // Returns the hash of an empty string if the tree has no leaves
  // (and hence, no root).
  virtual std::string CurrentRoot();

 private:
  // Append a node to the level.
  void PushBack(size_t level, std::string node);

  void UpdateRoot();
  // Since the tree is append-only to the right, at any given point in time,
  // at each level, all nodes that have a right sibling are fixed and will
  // no longer change. Thus we store, for each level i, only the last lone
  // left node (if one exists) or an empty string otherwise (tree_[i]).
  //
  //        ___hash___
  //       |          |
  //    __ h20__    __.
  //   |       |   |
  //  h10     h11  .
  //  | |     | |  |
  // a0 a1   a2 a3 a4
  //
  // is internally represented, top-down
  //
  // --------
  // | h20  |      tree_[2]
  // --------
  // |      |      tree_[1]
  // --------
  // |  a4  |      tree_[0]
  // --------
  //
  // After adding a 6th hash a5, the tree becomes
  //
  //        ___ hash'___
  //       |            |
  //    __ h20__     __.
  //   |        |   |
  //  h10     h11   h12
  //  | |     | |   | |
  // a0 a1   a2 a3 a4 a5
  //
  // and its internal representation is
  //
  // --------
  // | h20  |      tree_[2]
  // --------
  // | h12  |      tree_[1]
  // --------
  // |      |      tree_[0]
  // --------

  std::vector<std::string> tree_;
  TreeHasher treehasher_;
  // True number of leaves in the tree.
  size_t leaf_count_;
  // Number of leaves propagated up to the root,
  // to keep track of lazy evaluation.
  size_t leaves_processed_;
  // True number of levels in the tree. Note that the tree_ itself
  // contains the root only if it's balanced, so tree_.size() does not always
  // match true level count.
  size_t level_count_;
  // The root for |leaves_processed_| leaves.
  std::string root_;
};
#endif
