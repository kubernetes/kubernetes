#ifndef MERKLE_TREE_MATH_H
#define MERKLE_TREE_MATH_H
#include <stddef.h>

class MerkleTreeMath {
 public:
  static bool IsPowerOfTwoPlusOne(size_t leaf_count);

  // Index of the parent node in the parent level of the tree.
  static size_t Parent(size_t leaf);

  // True if the node is a right child; false if it is the left (or only)
  // child.
  static bool IsRightChild(size_t leaf);

  // Index of the node's (left or right) sibling in the same level.
  static size_t Sibling(size_t leaf);

 private:
  MerkleTreeMath();
};
#endif
