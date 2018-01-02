#ifndef MERKLEVERIFIER_H
#define MERKLEVERIFIER_H

#include <stddef.h>
#include <vector>

#include "merkletree/tree_hasher.h"

class SerialHasher;

// Class for verifying paths emitted by MerkleTrees.
// TODO: consistency proofs between snapshots.

class MerkleVerifier {
 public:
  // Takes ownership of the SerialHasher.
  MerkleVerifier(SerialHasher* hasher);
  ~MerkleVerifier();

  // Verify Merkle path. Return true iff the path is a valid proof for
  // the leaf in the tree, i.e., iff 0 < leaf <= tree_size and path
  // is a valid path from the leaf hash of data to the root.
  //
  // @param leaf index of the leaf.
  // @param tree_size number of leaves in the tree.
  // @param path a vector of node hashes ordered according to levels from leaf
  // to root. Does not include the leaf hash or the root.
  // @ param root The root hash
  // @ param data The leaf data
  bool VerifyPath(size_t leaf, size_t tree_size,
                  const std::vector<std::string>& path,
                  const std::string& root, const std::string& data);

  // Compute the root corresponding to a Merkle audit path.
  // Returns an empty string if the path is not valid.
  //
  // @param leaf index of the leaf.
  // @param tree_size number of leaves in the tree.
  // @param path a vector of node hashes ordered according to levels from leaf
  // to root. Does not include the leaf hash or the root.
  // @ param data The leaf data
  std::string RootFromPath(size_t leaf, size_t tree_size,
                           const std::vector<std::string>& path,
                           const std::string& data);

  bool VerifyConsistency(size_t snapshot1, size_t snapshot2,
                         const std::string& root1, const std::string& root2,
                         const std::vector<std::string>& proof);

  // Return the leaf hash corresponding to the leaf input.
  std::string LeafHash(const std::string& data);

 private:
  TreeHasher treehasher_;
};

#endif
