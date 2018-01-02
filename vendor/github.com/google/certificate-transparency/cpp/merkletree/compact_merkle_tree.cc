#include "merkletree/compact_merkle_tree.h"

#include <assert.h>
#include <stddef.h>
#include <string>
#include <vector>

#include "merkletree/merkle_tree_math.h"

using cert_trans::MerkleTreeInterface;
using std::string;

CompactMerkleTree::CompactMerkleTree(SerialHasher* hasher)
    : MerkleTreeInterface(),
      treehasher_(hasher),
      leaf_count_(0),
      leaves_processed_(0),
      level_count_(0),
      root_(treehasher_.HashEmpty()) {
}

CompactMerkleTree::CompactMerkleTree(MerkleTree& model, SerialHasher* hasher)
    : MerkleTreeInterface(),
      tree_(std::max<int64_t>(0, model.LevelCount() - 1)),
      treehasher_(hasher),
      leaf_count_(model.LeafCount()),
      leaves_processed_(0),
      level_count_(model.LevelCount()),
      root_(treehasher_.HashEmpty()) {
  if (model.LeafCount() == 0) {
    return;
  }
  // Get the inclusion proof path to the last entry in the tree, which by
  // definition must consist purely of left-hand nodes.
  std::vector<string> path(model.PathToCurrentRoot(model.LeafCount()));
  if (!path.empty()) {
    /* We have to do some juggling here as tree_[] differs from our MerkleTree
    // structure in that incomplete right-hand subtrees 'fall-through' to lower
    // levels:
    //
    // MerkleTree structure for 3 leaves:
    //      R
    //     / \
    //    /   \
    //   AB    c
    //  / \
    // a   b
    //
    // Compact tree represents this as:
    //      R
    //     / \
    //    /   \
    //   AB    .
    //         |
    //         c
    // or:
    // tree_[1] = AB
    // tree_[0] = c  // (c) has "fallen-through" to the lowest level
    //
    // The inclusion proof path for the right-most entry effectively
    // describes the state of the tree immediately before the right-most
    // entry was added.
    // Since the inclusion proof path consists exclusively of left-hand
    // nodes and each entry in the path covers the maximum sub-tree possible,
    // we can use this to directly construct the Compact respresentation of
    // the tree before the newest entry was added.
    */

    // index into tree_, starting at the leaf level:
    int level(0);
    std::vector<string>::const_iterator i = path.begin();
    size_t size_of_previous_tree(model.LeafCount() - 1);
    for (; size_of_previous_tree != 0; size_of_previous_tree >>= 1) {
      if ((size_of_previous_tree & 1) != 0) {
        // if the level'th bit in the previous tree size is set, then we have
        // a proof path entry for this level (because proof entries cover the
        // maximum possible sub-tree.)
        tree_[level] = *i;
        i++;
      }
      level++;
    }
    assert(i == path.end());
  }

  // Now tree_ should contain a representation of the tree state just before
  // the last entry was added, so we PushBack the final right-hand entry
  // here, which will perform any recalculations necessary to reach the final
  // tree.
  PushBack(0, model.LeafHash(model.LeafCount()));
  assert(model.CurrentRoot() == CurrentRoot());
  assert(model.LeafCount() == LeafCount());
  assert(model.LevelCount() == LevelCount());
}


CompactMerkleTree::CompactMerkleTree(const CompactMerkleTree& other,
                                     SerialHasher* hasher)
    : tree_(other.tree_),
      treehasher_(hasher),
      leaf_count_(other.leaf_count_),
      leaves_processed_(other.leaves_processed_),
      level_count_(other.level_count_),
      root_(other.root_) {
}

CompactMerkleTree::~CompactMerkleTree() {
}


size_t CompactMerkleTree::AddLeaf(const string& data) {
  return AddLeafHash(treehasher_.HashLeaf(data));
}

size_t CompactMerkleTree::AddLeafHash(const string& hash) {
  PushBack(0, hash);
  // Update level count: a k-level tree can hold 2^{k-1} leaves,
  // so increment level count every time we overflow a power of two.
  // Do not update the root; we evaluate the tree lazily.
  if (MerkleTreeMath::IsPowerOfTwoPlusOne(++leaf_count_))
    ++level_count_;
  return leaf_count_;
}

string CompactMerkleTree::CurrentRoot() {
  UpdateRoot();
  return root_;
}

void CompactMerkleTree::PushBack(size_t level, string node) {
  assert(node.size() == treehasher_.DigestSize());
  if (tree_.size() <= level) {
    // First node at a new level.
    tree_.push_back(node);
  } else if (tree_[level].empty()) {
    // Lone left sibling.
    tree_[level] = node;
  } else {
    // Left sibling waiting: hash together and propagate up.
    PushBack(level + 1, treehasher_.HashChildren(tree_[level], node));
    tree_[level].clear();
  }
}

void CompactMerkleTree::UpdateRoot() {
  if (leaves_processed_ == LeafCount())
    return;

  string right_sibling;

  for (size_t level = 0; level < tree_.size(); ++level) {
    if (!tree_[level].empty()) {
      // A lonely left sibling gets pulled up as a right sibling.
      if (right_sibling.empty())
        right_sibling = tree_[level];
      else
        right_sibling = treehasher_.HashChildren(tree_[level], right_sibling);
    }
  }

  root_ = right_sibling;
  leaves_processed_ = LeafCount();
}
