#include <gtest/gtest.h>
#include <stddef.h>
#include <string>

#include "merkletree/serial_hasher.h"
#include "merkletree/tree_hasher.h"
#include "util/testing.h"
#include "util/util.h"

namespace {

using std::string;

typedef struct {
  size_t input_length;
  const char* input;
  const char* output;
} LeafTestVector;

// Inputs and outputs are of fixed digest size.
typedef struct {
  const char* left;
  const char* right;
  const char* output;
} NodeTestVector;

const char sha256_empty_hash[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

LeafTestVector sha256_leaves[] = {
    {0, "",
     "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"},
    {1, "00",
     "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"},

    {16, "101112131415161718191a1b1c1d1e1f",
     "3bfb960453ebaebf33727da7a1f4db38acc051d381b6da20d6d4e88f0eabfd7a"},
    {0, NULL, NULL}};

NodeTestVector sha256_nodes[] = {
    {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
     "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
     "1a378704c17da31e2d05b6d121c2bb2c7d76f6ee6fa8f983e596c2d034963c57"},
    {NULL, NULL, NULL}};

typedef struct {
  const char* empty_hash;
  LeafTestVector* leaves;
  NodeTestVector* nodes;
} TestVector;

TestVector test_sha256 = {sha256_empty_hash, sha256_leaves, sha256_nodes};

// A slightly shorter notation for constructing binary blobs from test vectors.
#define S(t, n) util::BinaryString(string((t), (2 * n)))
// The reverse
#define H(t) util::HexString(t)

template <class T>
TestVector* TestVectors();

template <>
TestVector* TestVectors<Sha256Hasher>() {
  return &test_sha256;
}

template <class T>
class TreeHasherTest : public ::testing::Test {
 protected:
  TreeHasher tree_hasher_;
  TestVector* test_vectors_;
  TreeHasherTest() : tree_hasher_(new T()), test_vectors_(TestVectors<T>()) {
  }
};

typedef ::testing::Types<Sha256Hasher> Hashers;

TYPED_TEST_CASE(TreeHasherTest, Hashers);

// TreeHashers are collision resistant when used correctly, i.e.,
// when HashChildren() is called on the (fixed-length) outputs of HashLeaf().
TYPED_TEST(TreeHasherTest, CollisionTest) {
  string leaf1_digest, leaf2_digest, node1_digest, node2_digest;

  const size_t digestsize = this->tree_hasher_.DigestSize();

  // Check that the empty hash is not the same as the hash of an empty leaf.
  leaf1_digest = this->tree_hasher_.HashEmpty();
  EXPECT_EQ(leaf1_digest.size(), digestsize);

  leaf2_digest = this->tree_hasher_.HashLeaf(string());
  EXPECT_EQ(leaf2_digest.size(), digestsize);

  EXPECT_NE(H(leaf1_digest), H(leaf2_digest));

  // Check that different leaves hash to different digests.
  const char hello[] = "Hello";
  const char world[] = "World";
  string leaf1(hello, 5);
  string leaf2(world, 5);
  leaf1_digest = this->tree_hasher_.HashLeaf(leaf1);
  EXPECT_EQ(leaf1_digest.size(), digestsize);

  leaf2_digest = this->tree_hasher_.HashLeaf(leaf2);
  EXPECT_EQ(leaf2_digest.size(), digestsize);

  EXPECT_NE(H(leaf1_digest), H(leaf2_digest));

  // Compute an intermediate node digest.
  node1_digest = this->tree_hasher_.HashChildren(leaf1_digest, leaf2_digest);
  EXPECT_EQ(node1_digest.size(), digestsize);

  // Check that this is not the same as a leaf hash of their concatenation.
  node2_digest = this->tree_hasher_.HashLeaf(leaf1_digest + leaf2_digest);
  EXPECT_EQ(node2_digest.size(), digestsize);

  EXPECT_NE(H(node1_digest), H(node2_digest));

  // Swap the order of nodes and check that the hash is different.
  node2_digest = this->tree_hasher_.HashChildren(leaf2_digest, leaf1_digest);
  EXPECT_EQ(node2_digest.size(), digestsize);

  EXPECT_NE(H(node1_digest), H(node2_digest));
}

TYPED_TEST(TreeHasherTest, TestVectors) {
  // The empty hash
  string digest = this->tree_hasher_.HashEmpty();
  EXPECT_STREQ(this->test_vectors_->empty_hash, H(digest).c_str());

  // Leaf hashes
  for (size_t i = 0; this->test_vectors_->leaves[i].input != NULL; ++i) {
    digest = this->tree_hasher_.HashLeaf(
        S(this->test_vectors_->leaves[i].input,
          this->test_vectors_->leaves[i].input_length));
    EXPECT_STREQ(this->test_vectors_->leaves[i].output, H(digest).c_str());
  }

  // Node hashes
  for (size_t i = 0; this->test_vectors_->nodes[i].left != NULL; ++i) {
    digest =
        this->tree_hasher_.HashChildren(S(this->test_vectors_->nodes[i].left,
                                          this->tree_hasher_.DigestSize()),
                                        S(this->test_vectors_->nodes[i].right,
                                          this->tree_hasher_.DigestSize()));
    EXPECT_STREQ(this->test_vectors_->nodes[i].output, H(digest).c_str());
  }
}

#undef S
#undef H

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
