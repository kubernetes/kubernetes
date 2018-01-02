#include <gtest/gtest.h>
#include <stddef.h>
#include <string>

#include "merkletree/serial_hasher.h"
#include "util/testing.h"
#include "util/util.h"

namespace {

using std::string;

const char kTestString[] = "Hello world!";
const size_t kTestStringLength = 12;

typedef struct {
  size_t input_length;
  const char* input;
  const char* output;
} HashTestVector;

// A couple of SHA-256 test vectors from http://csrc.nist.gov/groups/STM/cavp/
HashTestVector test_sha256[] = {
    {0, "",
     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    {8, "5738c929c4f4ccb6",
     "963bb88f27f512777aab6c8b1a02c70ec0ad651d428f870036e1917120fb48bf"},
    {63,
     "e2f76e97606a872e317439f1a03fcd92e632e5bd4e7cbc4e97f1afc19a16fde9"
     "2d77cbe546416b51640cddb92af996534dfd81edb17c4424cf1ac4d75aceeb",
     "18041bd4665083001fba8c5411d2d748e8abbfdcdfd9218cb02b68a78e7d4c23"},
    // to indicate the end
    {0, NULL, NULL}};

// A slightly shorter notation for constructing binary blobs from test vectors.
#define S(t, n) util::BinaryString(string((t), (2 * n)))
// The reverse
#define H(t) util::HexString(t)

template <class T>
HashTestVector* TestVectors();

template <>
HashTestVector* TestVectors<Sha256Hasher>() {
  return test_sha256;
}

template <class T>
class SerialHasherTest : public ::testing::Test {
 protected:
  SerialHasher* hasher_;
  HashTestVector* test_vectors_;

  SerialHasherTest() : hasher_(new T()), test_vectors_(TestVectors<T>()) {
  }

  ~SerialHasherTest() {
    delete hasher_;
  }
};

typedef ::testing::Types<Sha256Hasher> Hashers;

TYPED_TEST_CASE(SerialHasherTest, Hashers);

// Known Answer Tests
TYPED_TEST(SerialHasherTest, TestVectors) {
  string input, output, digest;

  for (size_t i = 0; this->test_vectors_[i].input != NULL; ++i) {
    this->hasher_->Reset();
    this->hasher_->Update(
        S(this->test_vectors_[i].input, this->test_vectors_[i].input_length));
    digest = this->hasher_->Final();
    EXPECT_STREQ(H(digest).c_str(), this->test_vectors_[i].output);
  }
}

// Test fragmented updates
TYPED_TEST(SerialHasherTest, Update) {
  string input(kTestString, kTestStringLength), output, digest;

  this->hasher_->Reset();
  this->hasher_->Update(input);
  digest = this->hasher_->Final();
  EXPECT_EQ(digest.size(), this->hasher_->DigestSize());

  // The same in two chunks
  this->hasher_->Reset();
  this->hasher_->Update(input.substr(0, kTestStringLength / 2));
  this->hasher_->Update(input.substr(kTestStringLength / 2));
  output = this->hasher_->Final();
  EXPECT_EQ(H(digest), H(output));
}

TYPED_TEST(SerialHasherTest, Create) {
  string input, output, digest;

  for (size_t i = 0; this->test_vectors_[i].input != NULL; ++i) {
    SerialHasher* new_hasher = this->hasher_->Create();
    new_hasher->Reset();
    new_hasher->Update(
        S(this->test_vectors_[i].input, this->test_vectors_[i].input_length));
    digest = new_hasher->Final();
    EXPECT_STREQ(H(digest).c_str(), this->test_vectors_[i].output);
    delete new_hasher;
  }
}

TEST(Sha256Test, StaticDigest) {
  string input, output, digest;

  for (size_t i = 0; test_sha256[i].input != NULL; ++i) {
    digest = Sha256Hasher::Sha256Digest(
        S(test_sha256[i].input, test_sha256[i].input_length));
    EXPECT_STREQ(H(digest).c_str(), test_sha256[i].output);
  }
}

#undef S
#undef H

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
