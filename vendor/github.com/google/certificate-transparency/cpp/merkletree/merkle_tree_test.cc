#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <vector>

#include "merkletree/compact_merkle_tree.h"
#include "merkletree/merkle_tree.h"
#include "merkletree/merkle_verifier.h"
#include "merkletree/serial_hasher.h"
#include "merkletree/tree_hasher.h"
#include "util/testing.h"
#include "util/util.h"

namespace {

using std::string;

// REFERENCE IMPLEMENTATIONS

// Get the largest power of two smaller than i.
int DownToPowerOfTwo(int i) {
  CHECK_GE(i, 2);
  // Find the smallest power of two greater than or equal to i.
  int split = 1;
  do {
    split <<= 1;
  } while (split < i);

  // Get the largest power of two smaller than i.
  return split >> 1;
}

// Reference implementation of Merkle hash, for cross-checking.
string ReferenceMerkleTreeHash(string inputs[], int input_size,
                               TreeHasher* treehasher) {
  if (!input_size)
    return treehasher->HashEmpty();
  if (input_size == 1)
    return treehasher->HashLeaf(inputs[0]);

  const int split = DownToPowerOfTwo(input_size);

  return treehasher->HashChildren(
      ReferenceMerkleTreeHash(&inputs[0], split, treehasher),
      ReferenceMerkleTreeHash(&inputs[split], input_size - split, treehasher));
}

// Reference implementation of Merkle paths. Path from leaf to root,
// excluding the leaf and root themselves.
std::vector<string> ReferenceMerklePath(string inputs[], int input_size,
                                        int leaf, TreeHasher* treehasher) {
  std::vector<string> path;
  if (leaf > input_size || leaf == 0)
    return path;

  if (input_size == 1)
    return path;

  const int split = DownToPowerOfTwo(input_size);

  std::vector<string> subpath;
  if (leaf <= split) {
    subpath = ReferenceMerklePath(&inputs[0], split, leaf, treehasher);
    path.insert(path.end(), subpath.begin(), subpath.end());
    path.push_back(ReferenceMerkleTreeHash(&inputs[split], input_size - split,
                                           treehasher));
  } else {
    subpath = ReferenceMerklePath(&inputs[split], input_size - split,
                                  leaf - split, treehasher);
    path.insert(path.end(), subpath.begin(), subpath.end());
    path.push_back(ReferenceMerkleTreeHash(&inputs[0], split, treehasher));
  }

  return path;
}

// Reference implementation of snapshot consistency.
// Call with have_root1 = true.
std::vector<string> ReferenceSnapshotConsistency(string inputs[],
                                                 int snapshot2, int snapshot1,
                                                 TreeHasher* treehasher,
                                                 bool have_root1) {
  std::vector<string> proof;
  if (snapshot1 == 0 || snapshot1 > snapshot2)
    return proof;
  if (snapshot1 == snapshot2) {
    // Consistency proof for two equal subtrees is empty.
    if (!have_root1)
      // Record the hash of this subtree unless it's the root for which
      // the proof was originally requested. (This happens when the snapshot1
      // tree is balanced.)
      proof.push_back(ReferenceMerkleTreeHash(inputs, snapshot1, treehasher));
    return proof;
  }

  // 0 < snapshot1 < snapshot2
  const int split = DownToPowerOfTwo(snapshot2);

  std::vector<string> subproof;
  if (snapshot1 <= split) {
    // Root of snapshot1 is in the left subtree of snapshot2.
    // Prove that the left subtrees are consistent.
    subproof = ReferenceSnapshotConsistency(inputs, split, snapshot1,
                                            treehasher, have_root1);
    proof.insert(proof.end(), subproof.begin(), subproof.end());
    // Record the hash of the right subtree (only present in snapshot2).
    proof.push_back(ReferenceMerkleTreeHash(&inputs[split], snapshot2 - split,
                                            treehasher));
  } else {
    // Snapshot1 root is at the same level as snapshot2 root.
    // Prove that the right subtrees are consistent. The right subtree
    // doesn't contain the root of snapshot1, so set have_root1 = false.
    subproof =
        ReferenceSnapshotConsistency(&inputs[split], snapshot2 - split,
                                     snapshot1 - split, treehasher, false);
    proof.insert(proof.end(), subproof.begin(), subproof.end());
    // Record the hash of the left subtree (equal in both trees).
    proof.push_back(ReferenceMerkleTreeHash(&inputs[0], split, treehasher));
  }
  return proof;
}

class MerkleTreeTest : public ::testing::Test {
 protected:
  TreeHasher tree_hasher_;
  std::vector<string> data_;
  MerkleTreeTest() : tree_hasher_(new Sha256Hasher()) {
    for (int i = 0; i < 256; ++i)
      data_.push_back(string(1, i));
  }
};

class MerkleTreeFuzzTest : public MerkleTreeTest {
 protected:
  MerkleTreeFuzzTest() : MerkleTreeTest() {
  }
  void SetUp() {
    srand(time(NULL));
  }
};

class CompactMerkleTreeTest : public MerkleTreeTest {};

class CompactMerkleTreeFuzzTest : public MerkleTreeFuzzTest {
 protected:
  string RandomLeaf(size_t len) {
    string r;
    for (size_t i = 0; i < len; ++i) {
      r += uint8_t(rand() % 0xff);
    }
    return r;
  }
};

// FUZZ TESTS AGAINST REFERENCE IMPLEMENTATIONS

// Make random root queries and check against the reference hash.
TEST_F(MerkleTreeFuzzTest, RootFuzz) {
  for (size_t tree_size = 1; tree_size <= data_.size(); ++tree_size) {
    MerkleTree tree(new Sha256Hasher());
    for (size_t j = 0; j < tree_size; ++j)
      tree.AddLeaf(data_[j]);
    // Since the tree is evaluated lazily, the order of queries is significant.
    // Generate a random sequence of 8 queries for each tree.
    for (size_t j = 0; j < 8; ++j) {
      // A snapshot in the range 0...tree_size.
      const size_t snapshot = rand() % (tree_size + 1);
      EXPECT_EQ(tree.RootAtSnapshot(snapshot),
                ReferenceMerkleTreeHash(data_.data(), snapshot,
                                        &tree_hasher_));
    }
  }
}

TEST_F(CompactMerkleTreeTest, RootFuzz) {
  for (size_t tree_size = 1; tree_size <= data_.size(); ++tree_size) {
    CompactMerkleTree tree(new Sha256Hasher());
    for (size_t j = 0; j < tree_size; ++j) {
      tree.AddLeaf(data_[j]);
      // Since the tree is evaluated lazily, the tree state is significant
      // when querying the root hash. Flip a coin and decide whether to
      // query now or later.
      if (rand() & 1) {
        EXPECT_EQ(tree.CurrentRoot(),
                  ReferenceMerkleTreeHash(data_.data(), j + 1, &tree_hasher_));
      }
    }
  }
}

// Make random path queries and check against the reference implementation.
TEST_F(MerkleTreeFuzzTest, PathFuzz) {
  for (size_t tree_size = 1; tree_size <= data_.size(); ++tree_size) {
    MerkleTree tree(new Sha256Hasher());
    for (size_t j = 0; j < tree_size; ++j)
      tree.AddLeaf(data_[j]);

    // Since the tree is evaluated lazily, the order of queries is significant.
    // Generate a random sequence of 8 queries for each tree.
    for (size_t j = 0; j < 8; ++j) {
      // A snapshot in the range 0... length.
      const size_t snapshot = rand() % (tree_size + 1);
      // A leaf in the range 0... snapshot.
      const size_t leaf = rand() % (snapshot + 1);
      EXPECT_EQ(tree.PathToRootAtSnapshot(leaf, snapshot),
                ReferenceMerklePath(data_.data(), snapshot, leaf,
                                    &tree_hasher_));
    }
  }
}

// Make random proof queries and check against the reference implementation.
TEST_F(MerkleTreeFuzzTest, ConsistencyFuzz) {
  for (size_t tree_size = 1; tree_size <= data_.size(); ++tree_size) {
    MerkleTree tree(new Sha256Hasher());
    for (size_t j = 0; j < tree_size; ++j)
      tree.AddLeaf(data_[j]);

    // Since the tree is evaluated lazily, the order of queries is significant.
    // Generate a random sequence of 8 queries for each tree.
    for (size_t j = 0; j < 8; ++j) {
      // A snapshot in the range 0... length.
      const size_t snapshot2 = rand() % (tree_size + 1);
      // A snapshot in the range 0... snapshot.
      const size_t snapshot1 = rand() % (snapshot2 + 1);
      EXPECT_EQ(tree.SnapshotConsistency(snapshot1, snapshot2),
                ReferenceSnapshotConsistency(data_.data(), snapshot2,
                                             snapshot1, &tree_hasher_, true));
    }
  }
}

// KNOWN ANSWER TESTS

typedef struct {
  const char* str;
  int length_bytes;
} TestVector;

// A slightly shorter notation for constructing binary blobs from test vectors.
#define S(t) util::BinaryString(string(t.str, 2 * t.length_bytes))
// The reverse
#define H(t) util::HexString(t)

// The hash of an empty tree is the hash of the empty string.
// (see SerialHasherTest and http://csrc.nist.gov/groups/STM/cavp/)
const TestVector kSHA256EmptyTreeHash = {
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 32};

// Inputs to the reference tree, which has eight leaves.
const TestVector kInputs[8] = {
    {"", 0},
    {"00", 1},
    {"10", 1},
    {"2021", 2},
    {"3031", 2},
    {"40414243", 4},
    {"5051525354555657", 8},
    {"606162636465666768696a6b6c6d6e6f", 16},
};

// Level counts for number of leaves in [1, 8]
const size_t kLevelCounts[8] = {1, 2, 3, 3, 4, 4, 4, 4};

// Incremental roots from building the reference tree from inputs leaf-by-leaf.
// Generated from ReferenceMerkleTreeHash.
const TestVector kSHA256Roots[8] = {
    {"6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d", 32},
    {"fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125", 32},
    {"aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77", 32},
    {"d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7", 32},
    {"4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4", 32},
    {"76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef", 32},
    {"ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c", 32},
    {"5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328", 32}};

TEST_F(MerkleTreeTest, RootTestVectors) {
  // The first tree: add nodes one by one.
  MerkleTree tree1(new Sha256Hasher());
  EXPECT_EQ(tree1.LeafCount(), 0U);
  EXPECT_EQ(tree1.LevelCount(), 0U);
  EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256EmptyTreeHash.str);
  for (size_t i = 0; i < 8; ++i) {
    tree1.AddLeaf(S(kInputs[i]));
    EXPECT_EQ(tree1.LeafCount(), i + 1);
    EXPECT_EQ(tree1.LevelCount(), kLevelCounts[i]);
    EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256Roots[i].str);
    EXPECT_STREQ(H(tree1.RootAtSnapshot(0)).c_str(), kSHA256EmptyTreeHash.str);
    for (size_t j = 0; j <= i; ++j) {
      EXPECT_STREQ(H(tree1.RootAtSnapshot(j + 1)).c_str(),
                   kSHA256Roots[j].str);
    }

    for (size_t j = i + 1; j < 8; ++j) {
      EXPECT_EQ(tree1.RootAtSnapshot(j + 1), string());
    }
  }

  // The second tree: add all nodes at once.
  MerkleTree tree2(new Sha256Hasher());
  for (int i = 0; i < 8; ++i) {
    tree2.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree2.LeafCount(), 8U);
  EXPECT_EQ(tree2.LevelCount(), kLevelCounts[7]);
  EXPECT_STREQ(H(tree2.CurrentRoot()).c_str(), kSHA256Roots[7].str);

  // The third tree: add nodes in two chunks.
  MerkleTree tree3(new Sha256Hasher());
  // Add three nodes.
  for (int i = 0; i < 3; ++i) {
    tree3.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree3.LeafCount(), 3U);
  EXPECT_EQ(tree3.LevelCount(), kLevelCounts[2]);
  EXPECT_STREQ(H(tree3.CurrentRoot()).c_str(), kSHA256Roots[2].str);
  // Add the remaining nodes.
  for (int i = 3; i < 8; ++i) {
    tree3.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree3.LeafCount(), 8U);
  EXPECT_EQ(tree3.LevelCount(), kLevelCounts[7]);
  EXPECT_STREQ(H(tree3.CurrentRoot()).c_str(), kSHA256Roots[7].str);
}

TEST_F(CompactMerkleTreeTest, RootTestVectors) {
  // The first tree: add nodes one by one.
  CompactMerkleTree tree1(new Sha256Hasher());
  EXPECT_EQ(tree1.LeafCount(), 0U);
  EXPECT_EQ(tree1.LevelCount(), 0U);
  EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256EmptyTreeHash.str);
  for (size_t i = 0; i < 8; ++i) {
    tree1.AddLeaf(S(kInputs[i]));
    EXPECT_EQ(tree1.LeafCount(), i + 1);
    EXPECT_EQ(tree1.LevelCount(), kLevelCounts[i]);
    EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256Roots[i].str);
  }

  // The second tree: add all nodes at once.
  CompactMerkleTree tree2(new Sha256Hasher());
  for (int i = 0; i < 8; ++i) {
    tree2.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree2.LeafCount(), 8U);
  EXPECT_EQ(tree2.LevelCount(), kLevelCounts[7]);
  EXPECT_STREQ(H(tree2.CurrentRoot()).c_str(), kSHA256Roots[7].str);

  // The third tree: add nodes in two chunks.
  CompactMerkleTree tree3(new Sha256Hasher());
  // Add three nodes.
  for (int i = 0; i < 3; ++i) {
    tree3.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree3.LeafCount(), 3U);
  EXPECT_EQ(tree3.LevelCount(), kLevelCounts[2]);
  EXPECT_STREQ(H(tree3.CurrentRoot()).c_str(), kSHA256Roots[2].str);
  // Add the remaining nodes.
  for (int i = 3; i < 8; ++i) {
    tree3.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree3.LeafCount(), 8U);
  EXPECT_EQ(tree3.LevelCount(), kLevelCounts[7]);
  EXPECT_STREQ(H(tree3.CurrentRoot()).c_str(), kSHA256Roots[7].str);
}

TEST_F(CompactMerkleTreeTest, TestCopyCtorWithRootTestVectors) {
  MerkleTree tree1(new Sha256Hasher());
  CompactMerkleTree refctree1(new Sha256Hasher());
  EXPECT_EQ(tree1.LeafCount(), 0U);
  EXPECT_EQ(tree1.LevelCount(), 0U);
  EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256EmptyTreeHash.str);
  for (size_t i = 0; i < 8; ++i) {
    tree1.AddLeaf(S(kInputs[i]));
    refctree1.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree1.LeafCount(), 8U);
  EXPECT_EQ(tree1.LevelCount(), kLevelCounts[7]);
  EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256Roots[7].str);

  CompactMerkleTree ctree1(tree1, new Sha256Hasher());
  EXPECT_EQ(tree1.LeafCount(), ctree1.LeafCount());
  EXPECT_EQ(tree1.LevelCount(), ctree1.LevelCount());
  EXPECT_EQ(tree1.CurrentRoot(), ctree1.CurrentRoot());
}

TEST_F(CompactMerkleTreeTest, TestCopyCtorThenAddLeafWithRootTestVectors) {
  MerkleTree tree(new Sha256Hasher());
  EXPECT_EQ(tree.LeafCount(), 0U);
  EXPECT_EQ(tree.LevelCount(), 0U);
  EXPECT_STREQ(H(tree.CurrentRoot()).c_str(), kSHA256EmptyTreeHash.str);
  for (size_t i = 0; i < 5; ++i) {
    tree.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree.LeafCount(), 5U);
  EXPECT_EQ(tree.LevelCount(), kLevelCounts[4]);
  EXPECT_STREQ(H(tree.CurrentRoot()).c_str(), kSHA256Roots[4].str);
  CompactMerkleTree ctree(tree, new Sha256Hasher());
  EXPECT_EQ(tree.LeafCount(), ctree.LeafCount());
  EXPECT_EQ(tree.LevelCount(), ctree.LevelCount());
  EXPECT_EQ(tree.CurrentRoot(), ctree.CurrentRoot());

  // Add the remaining nodes.
  for (int i = 5; i < 6; ++i) {
    tree.AddLeaf(S(kInputs[i]));
    ctree.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree.LeafCount(), 6U);
  EXPECT_EQ(tree.LevelCount(), kLevelCounts[5]);
  EXPECT_STREQ(H(tree.CurrentRoot()).c_str(), kSHA256Roots[5].str);
  EXPECT_EQ(tree.LeafCount(), ctree.LeafCount());
  EXPECT_EQ(tree.LevelCount(), ctree.LevelCount());
  EXPECT_EQ(tree.CurrentRoot(), ctree.CurrentRoot());
}

TEST_F(CompactMerkleTreeFuzzTest, CopyCtorForLargerTreesThenAppend) {
  for (size_t tree_size = 1; tree_size <= 512; ++tree_size) {
    // Build a tree of |tree_size| from random leaves
    MerkleTree tree(new Sha256Hasher());
    for (size_t i = 0; i < tree_size; ++i) {
      const string l(RandomLeaf(256));
      tree.AddLeaf(l);
    }
    EXPECT_EQ(tree.LeafCount(), tree_size);
    // Now build a CompactMerkleTree using |tree| as the model
    CompactMerkleTree ctree(tree, new Sha256Hasher());
    // And check that the public interface concurs
    EXPECT_EQ(tree.LeafCount(), ctree.LeafCount());
    EXPECT_EQ(tree.LevelCount(), ctree.LevelCount());
    EXPECT_EQ(tree.CurrentRoot(), ctree.CurrentRoot());
    // Now add a bunch more nodes and check that the compact tree
    // doesn't diverge from the reference tree:
    for (size_t i = 0; i < 256; ++i) {
      const string l(RandomLeaf(256));
      tree.AddLeaf(l);
      ctree.AddLeaf(l);
      EXPECT_EQ(tree.LeafCount(), ctree.LeafCount());
      EXPECT_EQ(tree.LevelCount(), ctree.LevelCount());
      EXPECT_EQ(tree.CurrentRoot(), ctree.CurrentRoot());
    }
  }
}

// Some paths for the reference tree.
typedef struct {
  int leaf;
  int snapshot;
  int path_length;
  TestVector path[3];
} PathTestVector;

// Generated from ReferenceMerklePath.
const PathTestVector kSHA256Paths[6] = {
    {0, 0, 0, {{"", 0}, {"", 0}, {"", 0}}},
    {1, 1, 0, {{"", 0}, {"", 0}, {"", 0}}},
    {1,
     8,
     3,
     {{"96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7", 32},
      {"5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e", 32},
      {"6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4",
       32}}},
    {6,
     8,
     3,
     {{"bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b", 32},
      {"ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0", 32},
      {"d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
       32}}},
    {3,
     3,
     1,
     {{"fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125", 32},
      {"", 0},
      {"", 0}}},
    {2,
     5,
     3,
     {{"6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d", 32},
      {"5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e", 32},
      {"bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b",
       32}}}};

TEST_F(MerkleTreeTest, PathTestVectors) {
  // First tree: build in one go.
  MerkleTree tree1(new Sha256Hasher());
  for (int i = 0; i < 8; ++i) {
    tree1.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree1.LeafCount(), 8U);
  EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256Roots[7].str);

  EXPECT_TRUE(tree1.PathToCurrentRoot(9).empty());
  for (int i = 0; i < 6; ++i) {
    std::vector<string> path =
        tree1.PathToRootAtSnapshot(kSHA256Paths[i].leaf,
                                   kSHA256Paths[i].snapshot);
    std::vector<string> kat_path;
    for (int j = 0; j < kSHA256Paths[i].path_length; ++j)
      kat_path.push_back(S(kSHA256Paths[i].path[j]));
    EXPECT_EQ(path, kat_path);
  }

  // Second tree: build incrementally.
  MerkleTree tree2(new Sha256Hasher());
  EXPECT_EQ(tree2.PathToCurrentRoot(0), tree1.PathToRootAtSnapshot(0, 0));
  EXPECT_TRUE(tree2.PathToCurrentRoot(1).empty());
  for (int i = 0; i < 8; ++i) {
    tree2.AddLeaf(S(kInputs[i]));
    for (int j = 0; j <= i + 1; ++j) {
      EXPECT_EQ(tree1.PathToRootAtSnapshot(j, i + 1),
                tree2.PathToCurrentRoot(j));
    }
    for (int j = i + 2; j <= 9; ++j)
      EXPECT_TRUE(tree1.PathToRootAtSnapshot(j, i + 1).empty());
  }
}

typedef struct {
  int snapshot1;
  int snapshot2;
  int proof_length;
  TestVector proof[3];
} ProofTestVector;

// Generated from ReferenceSnapshotConsistency.
const ProofTestVector kSHA256Proofs[4] = {
    {1, 1, 0, {{"", 0}, {"", 0}, {"", 0}}},
    {1,
     8,
     3,
     {{"96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7", 32},
      {"5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e", 32},
      {"6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4",
       32}}},
    {6,
     8,
     3,
     {{"0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a", 32},
      {"ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0", 32},
      {"d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
       32}}},
    {2,
     5,
     2,
     {{"5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e", 32},
      {"bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b", 32},
      {"", 0}}}};

TEST_F(MerkleTreeTest, ConsistencyTestVectors) {
  MerkleTree tree1(new Sha256Hasher());
  for (int i = 0; i < 8; ++i) {
    tree1.AddLeaf(S(kInputs[i]));
  }
  EXPECT_EQ(tree1.LeafCount(), 8U);
  EXPECT_STREQ(H(tree1.CurrentRoot()).c_str(), kSHA256Roots[7].str);

  for (int i = 0; i < 4; ++i) {
    std::vector<string> proof =
        tree1.SnapshotConsistency(kSHA256Proofs[i].snapshot1,
                                  kSHA256Proofs[i].snapshot2);
    std::vector<string> kat_proof;
    for (int j = 0; j < kSHA256Proofs[i].proof_length; ++j)
      kat_proof.push_back(S(kSHA256Proofs[i].proof[j]));
    EXPECT_EQ(proof, kat_proof);
  }
}

TEST_F(MerkleTreeTest, AddLeafHash) {
  const char* kHashValue = "0123456789abcdef0123456789abcdef";
  MerkleTree tree(new Sha256Hasher());
  size_t index = tree.AddLeafHash(kHashValue);
  EXPECT_EQ(1U, index);
  EXPECT_EQ(kHashValue, tree.LeafHash(index));
}

TEST_F(CompactMerkleTreeTest, TestCloneEmptyTreeProducesWorkingTree) {
  MerkleTree tree(new Sha256Hasher);
  CompactMerkleTree compact(tree, new Sha256Hasher);
  EXPECT_STREQ(H(compact.CurrentRoot()).c_str(), kSHA256EmptyTreeHash.str);
}

// VERIFICATION TESTS

class MerkleVerifierTest : public MerkleTreeTest {
 protected:
  MerkleVerifier verifier_;
  MerkleVerifierTest() : MerkleTreeTest(), verifier_(new Sha256Hasher()) {
  }

  void VerifierCheck(int leaf, int tree_size, const std::vector<string>& path,
                     const string& root, const string& data) {
    // Verify the original path.
    EXPECT_EQ(H(verifier_.RootFromPath(leaf, tree_size, path, data)), H(root));
    EXPECT_TRUE(verifier_.VerifyPath(leaf, tree_size, path, root, data));

    // Wrong leaf index.
    EXPECT_FALSE(verifier_.VerifyPath(leaf - 1, tree_size, path, root, data));
    EXPECT_FALSE(verifier_.VerifyPath(leaf + 1, tree_size, path, root, data));
    EXPECT_FALSE(verifier_.VerifyPath(leaf ^ 2, tree_size, path, root, data));

    // Wrong tree height.
    EXPECT_FALSE(verifier_.VerifyPath(leaf, tree_size * 2, path, root, data));
    EXPECT_FALSE(verifier_.VerifyPath(leaf, tree_size / 2, path, root, data));

    // Wrong leaf.
    const char wrong_leaf[] = "WrongLeaf";
    EXPECT_FALSE(verifier_.VerifyPath(leaf, tree_size, path, root,
                                      string(wrong_leaf, 9)));

    // Wrong root.
    EXPECT_FALSE(verifier_.VerifyPath(leaf, tree_size, path,
                                      S(kSHA256EmptyTreeHash), data));

    // Wrong paths.
    std::vector<string> wrong_path;

    // Modify a single element on the path.
    for (size_t j = 0; j < path.size(); ++j) {
      wrong_path = path;
      wrong_path[j] = S(kSHA256EmptyTreeHash);
      EXPECT_FALSE(
          verifier_.VerifyPath(leaf, tree_size, wrong_path, root, data));
    }

    // Add garbage at the end of the path.
    wrong_path = path;
    wrong_path.push_back(string());
    EXPECT_FALSE(
        verifier_.VerifyPath(leaf, tree_size, wrong_path, root, data));
    wrong_path.pop_back();

    wrong_path.push_back(root);
    EXPECT_FALSE(
        verifier_.VerifyPath(leaf, tree_size, wrong_path, root, data));
    wrong_path.pop_back();

    // Remove a node from the end.
    if (!wrong_path.empty()) {
      wrong_path.pop_back();
      EXPECT_FALSE(
          verifier_.VerifyPath(leaf, tree_size, wrong_path, root, data));
    }

    // Add garbage in the beginning of the path.
    wrong_path.clear();
    wrong_path.push_back(string());
    wrong_path.insert(wrong_path.end(), path.begin(), path.end());
    EXPECT_FALSE(
        verifier_.VerifyPath(leaf, tree_size, wrong_path, root, data));

    wrong_path[0] = root;
    EXPECT_FALSE(
        verifier_.VerifyPath(leaf, tree_size, wrong_path, root, data));
  }

  void VerifierConsistencyCheck(int snapshot1, int snapshot2,
                                const string& root1, const string& root2,
                                const std::vector<string>& proof) {
    // Verify the original consistency proof.
    EXPECT_TRUE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1, root2,
                                            proof));

    if (proof.empty())
      // For simplicity test only non-trivial proofs that have root1 != root2
      // snapshot1 != 0 and snapshot1 != snapshot2.
      return;

    // Wrong snapshot index.
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1 - 1, snapshot2, root1,
                                             root2, proof));
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1 + 1, snapshot2, root1,
                                             root2, proof));
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1 ^ 2, snapshot2, root1,
                                             root2, proof));

    // Wrong tree height.
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2 * 2, root1,
                                             root2, proof));
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2 / 2, root1,
                                             root2, proof));

    // Wrong root.
    const char wrong_root[] = "WrongRoot";
    const string bwrong_root(wrong_root, 9);
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                             bwrong_root, proof));
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, bwrong_root,
                                             root2, proof));
    // Swap roots.
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root2,
                                             root1, proof));

    // Wrong proofs.
    std::vector<string> wrong_proof;
    // Empty proof.
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                             root2, wrong_proof));

    // Modify a single element in the proof.
    for (size_t j = 0; j < proof.size(); ++j) {
      wrong_proof = proof;
      wrong_proof[j] = S(kSHA256EmptyTreeHash);
      EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                               root2, wrong_proof));
    }

    // Add garbage at the end of the proof.
    wrong_proof = proof;
    wrong_proof.push_back(string());
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                             root2, wrong_proof));
    wrong_proof.pop_back();

    wrong_proof.push_back(proof.back());
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                             root2, wrong_proof));
    wrong_proof.pop_back();

    // Remove a node from the end.
    wrong_proof.pop_back();
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                             root2, wrong_proof));

    // Add garbage in the beginning of the proof.
    wrong_proof.clear();
    wrong_proof.push_back(string());
    wrong_proof.insert(wrong_proof.end(), proof.begin(), proof.end());
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                             root2, wrong_proof));

    wrong_proof[0] = proof[0];
    EXPECT_FALSE(verifier_.VerifyConsistency(snapshot1, snapshot2, root1,
                                             root2, wrong_proof));
  }
};

TEST_F(MerkleVerifierTest, VerifyPath) {
  std::vector<string> path;
  // Various invalid paths.
  EXPECT_FALSE(verifier_.VerifyPath(0, 0, path, string(), string()));
  EXPECT_FALSE(verifier_.VerifyPath(0, 1, path, string(), string()));
  EXPECT_FALSE(verifier_.VerifyPath(1, 0, path, string(), string()));
  EXPECT_FALSE(verifier_.VerifyPath(2, 1, path, string(), string()));

  EXPECT_FALSE(
      verifier_.VerifyPath(0, 0, path, S(kSHA256EmptyTreeHash), string()));
  EXPECT_FALSE(
      verifier_.VerifyPath(0, 1, path, S(kSHA256EmptyTreeHash), string()));
  EXPECT_FALSE(
      verifier_.VerifyPath(1, 0, path, S(kSHA256EmptyTreeHash), string()));
  EXPECT_FALSE(
      verifier_.VerifyPath(2, 1, path, S(kSHA256EmptyTreeHash), string()));

  // Known good paths.
  // i = 0 is an invalid path.
  for (int i = 1; i < 6; ++i) {
    // Construct the path.
    path.clear();
    for (int j = 0; j < kSHA256Paths[i].path_length; ++j)
      path.push_back(S(kSHA256Paths[i].path[j]));
    VerifierCheck(kSHA256Paths[i].leaf, kSHA256Paths[i].snapshot, path,
                  S(kSHA256Roots[kSHA256Paths[i].snapshot - 1]),
                  S(kInputs[kSHA256Paths[i].leaf - 1]));
  }

  // More tests with reference path generator.
  string root;
  for (size_t tree_size = 1; tree_size <= data_.size() / 2; ++tree_size) {
    // Repeat for each leaf in range.
    for (size_t leaf = 1; leaf <= tree_size; ++leaf) {
      path = ReferenceMerklePath(data_.data(), tree_size, leaf, &tree_hasher_);
      root = ReferenceMerkleTreeHash(data_.data(), tree_size, &tree_hasher_);
      VerifierCheck(leaf, tree_size, path, root, data_[leaf - 1]);
    }
  }
}

TEST_F(MerkleVerifierTest, VerifyConsistencyProof) {
  std::vector<string> proof;
  string root1, root2;
  // Snapshots that are always consistent.
  EXPECT_TRUE(verifier_.VerifyConsistency(0, 0, root1, root2, proof));
  EXPECT_TRUE(verifier_.VerifyConsistency(0, 1, root1, root2, proof));
  EXPECT_TRUE(verifier_.VerifyConsistency(1, 1, root1, root2, proof));

  // Invalid consistency proofs.
  // Time travel to the past.
  EXPECT_FALSE(verifier_.VerifyConsistency(1, 0, root1, root2, proof));
  EXPECT_FALSE(verifier_.VerifyConsistency(2, 1, root1, root2, proof));
  // Empty proof.
  EXPECT_FALSE(verifier_.VerifyConsistency(1, 2, root1, root2, proof));

  root1 = S(kSHA256EmptyTreeHash);
  // Roots don't match.
  EXPECT_FALSE(verifier_.VerifyConsistency(0, 0, root1, root2, proof));
  EXPECT_FALSE(verifier_.VerifyConsistency(1, 1, root1, root2, proof));
  // Roots match but the proof is not empty.
  root2 = S(kSHA256EmptyTreeHash);
  proof.push_back(S(kSHA256EmptyTreeHash));
  EXPECT_FALSE(verifier_.VerifyConsistency(0, 0, root1, root2, proof));
  EXPECT_FALSE(verifier_.VerifyConsistency(0, 1, root1, root2, proof));
  EXPECT_FALSE(verifier_.VerifyConsistency(1, 1, root1, root2, proof));

  // Known good proofs.
  for (int i = 0; i < 4; ++i) {
    proof.clear();
    for (int j = 0; j < kSHA256Proofs[i].proof_length; ++j)
      proof.push_back(S(kSHA256Proofs[i].proof[j]));
    const int snapshot1 = kSHA256Proofs[i].snapshot1;
    const int snapshot2 = kSHA256Proofs[i].snapshot2;
    VerifierConsistencyCheck(snapshot1, snapshot2,
                             S(kSHA256Roots[snapshot1 - 1]),
                             S(kSHA256Roots[snapshot2 - 1]), proof);
  }

  // More tests with reference proof generator.
  for (size_t tree_size = 1; tree_size <= data_.size() / 2; ++tree_size) {
    root2 = ReferenceMerkleTreeHash(data_.data(), tree_size, &tree_hasher_);
    // Repeat for each snapshot in range.
    for (size_t snapshot = 1; snapshot <= tree_size; ++snapshot) {
      proof = ReferenceSnapshotConsistency(data_.data(), tree_size, snapshot,
                                           &tree_hasher_, true);
      root1 = ReferenceMerkleTreeHash(data_.data(), snapshot, &tree_hasher_);
      VerifierConsistencyCheck(snapshot, tree_size, root1, root2, proof);
    }
  }
}

#undef S
#undef H

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
