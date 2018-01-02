#include "merkletree/merkle_tree_math.h"

#include <stddef.h>

// static
bool MerkleTreeMath::IsPowerOfTwoPlusOne(size_t leaf_count) {
  if (leaf_count == 0)
    return false;
  if (leaf_count == 1)
    return true;
  // leaf_count is a power of two plus one if and only if
  // ((leaf_count -1) & (leaf_count - 2)) has no bits set.
  return (((leaf_count - 1) & (leaf_count - 2)) == 0);
}

// Index of the parent node in the parent level of the tree.
size_t MerkleTreeMath::Parent(size_t leaf) {
  return leaf >> 1;
}

// True if the node is a right child; false if it is the left (or only) child.
bool MerkleTreeMath::IsRightChild(size_t leaf) {
  return leaf & 1;
}

// Index of the node's (left or right) sibling in the same level.
size_t MerkleTreeMath::Sibling(size_t leaf) {
  return IsRightChild(leaf) ? (leaf - 1) : (leaf + 1);
}
