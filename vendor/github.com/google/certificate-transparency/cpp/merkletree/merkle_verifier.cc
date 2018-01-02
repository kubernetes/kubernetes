#include "merkletree/merkle_verifier.h"

#include <stddef.h>
#include <vector>

using std::string;

MerkleVerifier::MerkleVerifier(SerialHasher* hasher) : treehasher_(hasher) {
}

MerkleVerifier::~MerkleVerifier() {
}

static inline size_t Parent(size_t leaf) {
  return leaf >> 1;
}

static inline bool IsRightChild(size_t leaf) {
  return leaf & 1;
}

bool MerkleVerifier::VerifyPath(size_t leaf, size_t tree_size,
                                const std::vector<string>& path,
                                const string& root, const string& data) {
  string path_root = RootFromPath(leaf, tree_size, path, data);
  if (path_root.empty())
    return false;
  return path_root == root;
}

string MerkleVerifier::RootFromPath(size_t leaf, size_t tree_size,
                                    const std::vector<string>& path,
                                    const string& data) {
  if (leaf > tree_size || leaf == 0)
    // No valid path exists.
    return string();

  size_t node = leaf - 1;
  size_t last_node = tree_size - 1;

  string node_hash = LeafHash(data);
  std::vector<string>::const_iterator it = path.begin();

  while (last_node) {
    if (it == path.end())
      // We've reached the end but we're not done yet.
      return string();
    if (IsRightChild(node))
      node_hash = treehasher_.HashChildren(*it++, node_hash);
    else if (node < last_node)
      node_hash = treehasher_.HashChildren(node_hash, *it++);
    // Else the sibling does not exist and the parent is a dummy copy.
    // Do nothing.

    node = Parent(node);
    last_node = Parent(last_node);
  }

  // Check that we've reached the end.
  if (it != path.end())
    return string();
  return node_hash;
}

bool MerkleVerifier::VerifyConsistency(size_t snapshot1, size_t snapshot2,
                                       const string& root1,
                                       const string& root2,
                                       const std::vector<string>& proof) {
  if (snapshot1 > snapshot2)
    // Can't go back in time.
    return false;
  if (snapshot1 == snapshot2)
    return root1 == root2 && proof.empty();
  if (snapshot1 == 0)
    // Any snapshot greater than 0 is consistent with snapshot 0.
    return proof.empty();
  // Now 0 < snapshot1 < snapshot2.
  // Verify the roots.
  size_t node = snapshot1 - 1;
  size_t last_node = snapshot2 - 1;
  if (proof.empty())
    return false;
  std::vector<string>::const_iterator it = proof.begin();
  // Move up until the first mutable node.
  while (IsRightChild(node)) {
    node = Parent(node);
    last_node = Parent(last_node);
  }

  string node1_hash;
  string node2_hash;
  if (node)
    node2_hash = node1_hash = *it++;
  else
    // The tree at snapshot1 was balanced, nothing to verify for root1.
    node2_hash = node1_hash = root1;
  while (node) {
    if (it == proof.end())
      return false;

    if (IsRightChild(node)) {
      node1_hash = treehasher_.HashChildren(*it, node1_hash);
      node2_hash = treehasher_.HashChildren(*it, node2_hash);
      ++it;
    } else if (node < last_node)
      // The sibling only exists in the later tree. The parent in the
      // snapshot1 tree is a dummy copy.
      node2_hash = treehasher_.HashChildren(node2_hash, *it++);
    // Else the sibling does not exist in either tree. Do nothing.

    node = Parent(node);
    last_node = Parent(last_node);
  }

  // Verify the first root.
  if (node1_hash != root1)
    return false;

  // Continue until the second root.
  while (last_node) {
    if (it == proof.end())
      // We've reached the end but we're not done yet.
      return false;

    node2_hash = treehasher_.HashChildren(node2_hash, *it++);
    last_node = Parent(last_node);
  }

  // Verify the second root.
  return node2_hash == root2 && it == proof.end();
}

string MerkleVerifier::LeafHash(const std::string& data) {
  return treehasher_.HashLeaf(data);
}
