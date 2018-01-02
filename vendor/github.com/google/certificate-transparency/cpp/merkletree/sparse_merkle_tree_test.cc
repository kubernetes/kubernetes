#include <glog/logging.h>
#include <gtest/gtest.h>
#include <openssl/bn.h>
#include <sys/resource.h>
#include <algorithm>
#include <map>
#include <random>
#include <string>

#include "merkletree/sparse_merkle_tree.h"
#include "util/openssl_scoped_types.h"
#include "util/testing.h"
#include "util/util.h"

namespace {

using cert_trans::ScopedBIGNUM;
using std::fill;
using std::lower_bound;
using std::map;
using std::mt19937;
using std::ostringstream;
using std::pair;
using std::random_device;
using std::reverse;
using std::string;
using std::to_string;
using std::unique_ptr;
using std::vector;
using util::ToBase64;


const char kEmptyRootHashB64[] =
    "xmifEIEqCYCXbZUz2Dh1KCFmFZVn7DUVVxbBQTr1PWo=";


struct KeyComp {
  const BIGNUM* AsBN(const ScopedBIGNUM& a) const {
    return a.get();
  }

  const BIGNUM* AsBN(const BIGNUM* a) const {
    return a;
  }

  const BIGNUM* AsBN(const pair<ScopedBIGNUM, string>& a) const {
    return a.first.get();
  }
};


struct KeyEq : public KeyComp {
  template <class A, class B>
  bool operator()(const A& a, const B& b) const {
    return BN_cmp(AsBN(a), AsBN(b)) == 0;
  }
};


struct KeyLess : public KeyComp {
  template <class A, class B>
  bool operator()(const A& a, const B& b) const {
    return BN_cmp(AsBN(a), AsBN(b)) < 0;
  }
};


typedef vector<pair<ScopedBIGNUM, string>> ValueList;

pair<ScopedBIGNUM, string> Value(uint64_t n, const string& v) {
  pair<ScopedBIGNUM, string> ret;
  ret.second = v;
  ret.first.reset(BN_new());
  BN_set_word(ret.first.get(), n);
  return ret;
}

// Implements (more-or-less) the reference python code given in the
// revocation transparency paper for calculating the root-hash of a sparse
// tree with a given set of leaf nodes.
class Reference {
 public:
  Reference(SerialHasher* hasher)
      : tree_hasher_(CHECK_NOTNULL(hasher)),
        hStarEmptyCache_{tree_hasher_.HashLeaf("")} {
  }

  // Calculates the root hash of a sparse merkle tree of depth |n|, containing
  // the leaf values in |values|.
  string HStar2(size_t n, ValueList* values) {
    // values should be sorted
    std::sort(values->begin(), values->end(), KeyLess());
    // and without dupes
    values->erase(std::unique(values->begin(), values->end(), KeyEq()),
                  values->end());
    // Sounds a lot like a map to me, but I've left it as a list because:
    // a) it's just reference code, and
    // b) I want it to be as similar as possible to the code in the paper.

    ScopedBIGNUM offset(BN_new());
    CHECK_EQ(1, BN_zero(offset.get()));
    const string ret(
        HStar2b(n, *values, values->begin(), values->end(), offset.get()));
    return ret;
  }


 private:
  // Calculates & caches the 'null' node at depth |n|
  string HStarEmpty(size_t n) {
    if (hStarEmptyCache_.size() <= n) {
      const string t(
          tree_hasher_.HashChildren(HStarEmpty(n - 1), HStarEmpty(n - 1)));
      CHECK_EQ(n, hStarEmptyCache_.size());
      hStarEmptyCache_.push_back(t);
    }
    CHECK_LT(n, hStarEmptyCache_.size());
    return hStarEmptyCache_[n];
  }

  // Calculates an internal subtree.
  string HStar2b(size_t n, const ValueList& values,
                 ValueList::const_iterator lo, ValueList::const_iterator hi,
                 BIGNUM* offset) {
    if (n == 0) {
      if (lo == hi) {
        // DIFF: return the null leaf hash, rather than "0" as in the paper.
        return hStarEmptyCache_[0];
      }
      CHECK_EQ(1, hi - lo);
      // DIFF: return H(\x00||value) rather than "1" as in the paper.
      return tree_hasher_.HashLeaf(lo->second);
    }
    if (lo == hi) {
      return HStarEmpty(n);
    }

    // DIFF: use BIGNUM, 'cos we'll get to values of O(1 << 256) here (!)
    ScopedBIGNUM split(BN_new());
    CHECK_EQ(1, BN_set_word(split.get(), 1));
    CHECK_EQ(1, BN_lshift(split.get(), split.get(), n - 1));
    CHECK_EQ(1, BN_add(split.get(), split.get(), offset));

    auto i(lower_bound(lo, hi, split, KeyLess()));
    const string ret(
        tree_hasher_.HashChildren(HStar2b(n - 1, values, lo, i, offset),
                                  HStar2b(n - 1, values, i, hi, split.get())));
    return ret;
  }

  TreeHasher tree_hasher_;
  vector<string> hStarEmptyCache_;
};


class SparseMerkleTreeTest : public testing::Test {
 public:
  SparseMerkleTreeTest()
      : tree_hasher_(new Sha256Hasher),
        tree_(new Sha256Hasher()),
        rand_({1234}) {
  }

 protected:
  // Returns a Path with the high 64 bits set to |high|
  SparseMerkleTree::Path PathHigh(uint64_t high) {
    SparseMerkleTree::Path ret;
    ret.fill(0);
    for (size_t i(0); i < 8; ++i) {
      ret[7 - i] = high & 0xff;
      high >>= 8;
    }
    return ret;
  }

  // Returns a Path with the low 64 bits set to |high|
  SparseMerkleTree::Path PathLow(uint64_t low) {
    SparseMerkleTree::Path ret;
    ret.fill(0);
    for (size_t i(0); i < 8; ++i) {
      ret[ret.size() - 1 - i] = low & 0xff;
      low >>= 8;
    }
    return ret;
  }

  // Returns a random Path.
  SparseMerkleTree::Path RandomPath() {
    SparseMerkleTree::Path ret;
    for (int i(0); i < ret.size(); ++i) {
      ret[i] = rand_() & 0xff;
    }
    return ret;
  }

  SparseMerkleTree::Path PathFromString(const string& s) {
    SparseMerkleTree::Path ret;
    CHECK_LE(s.size(), ret.size());
    fill(copy(s.begin(), s.end(), ret.begin()), ret.end(), 0);
    return ret;
  }

  TreeHasher tree_hasher_;
  SparseMerkleTree tree_;
  mt19937 rand_;
};


TEST_F(SparseMerkleTreeTest, PathBitAndPathStreamOperatorAgree) {
  ostringstream os;
  const SparseMerkleTree::Path p(RandomPath());
  os << p;
  string b;
  for (size_t i(0); i < SparseMerkleTree::kDigestSizeBits; ++i) {
    b += PathBit(p, i) == 0 ? '0' : '1';
  }
  EXPECT_EQ(os.str(), b);
}


TEST_F(SparseMerkleTreeTest, ReferenceEmptyTreeRootKAT) {
  Reference ref(new Sha256Hasher);
  ValueList empty_values;
  EXPECT_EQ(kEmptyRootHashB64, ToBase64(ref.HStar2(256, &empty_values)));
}


TEST_F(SparseMerkleTreeTest, EmptyTreeRootKAT) {
  tree_.CurrentRoot();
  EXPECT_EQ(kEmptyRootHashB64, ToBase64(tree_.CurrentRoot()));
}


TEST_F(SparseMerkleTreeTest, SimpleTest) {
  Reference ref(new Sha256Hasher);
  ValueList values;
  for (auto r : vector<uint64_t>{1, 5, 10}) {
    const string value(to_string(r));
    values.emplace_back(std::move(Value(r, value)));
    SparseMerkleTree::Path p(PathLow(r));
    tree_.SetLeaf(p, value);
  }
  const string ref_root(
      ref.HStar2(SparseMerkleTree::kDigestSizeBits, &values));
  const string smt_root(tree_.CurrentRoot());
  EXPECT_EQ(ToBase64(ref_root), ToBase64(smt_root));
}


TEST_F(SparseMerkleTreeTest, RandomReferenceTest) {
  Reference ref(new Sha256Hasher);
  ValueList values;
  LOG(INFO) << "Setup";
  for (int i(0); i < 10000; ++i) {
    uint64_t r(rand_() + i);
    const string value(to_string(r));
    values.emplace_back(std::move(Value(r, value)));
    const SparseMerkleTree::Path p(PathLow(r));
    tree_.SetLeaf(p, value);
  }
  LOG(INFO) << "Calculating SMT Root";
  const string smt_root(tree_.CurrentRoot());
  LOG(INFO) << "Calculating Reference Root";
  const string ref_root(ref.HStar2(256, &values));
  LOG(INFO) << "Comparing";
  EXPECT_EQ(ToBase64(ref_root), ToBase64(smt_root));
}


TEST_F(SparseMerkleTreeTest, DISABLED_RefMemTest) {
  Reference ref(new Sha256Hasher);
  ValueList values;

  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  long max_rss_before = ru.ru_maxrss;
  uint64_t time_before = util::TimeInMilliseconds();
  LOG(INFO) << "Setup";

  for (int i(0); i < 10000000; ++i) {
    uint64_t r(rand_() + i);
    const string value(to_string(r));
    values.emplace_back(std::move(Value(r, value)));
  }
  LOG(INFO) << "Calculating Root";
  const string ref_root(ref.HStar2(256, &values));
  LOG(INFO) << "Done";

  uint64_t time_after = util::TimeInMilliseconds();
  getrusage(RUSAGE_SELF, &ru);
  LOG(INFO) << "Peak RSS delta (as reported by getrusage()) was "
            << ru.ru_maxrss - max_rss_before << " kB";
  LOG(INFO) << "Elapsed time: " << time_after - time_before << " ms";
}


TEST_F(SparseMerkleTreeTest, DISABLED_SMTMemTest) {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  long max_rss_before = ru.ru_maxrss;
  uint64_t time_before = util::TimeInMilliseconds();
  LOG(INFO) << "Setup";

  for (int i(0); i < 10000000; ++i) {
    uint64_t r(rand_() + i);
    const string value(to_string(r));
    const SparseMerkleTree::Path p(PathLow(r));
    tree_.SetLeaf(p, value);
  }
  LOG(INFO) << "Calculating Root";
  const string smt_root(tree_.CurrentRoot());
  LOG(INFO) << "Done";

  uint64_t time_after = util::TimeInMilliseconds();
  getrusage(RUSAGE_SELF, &ru);
  LOG(INFO) << "Peak RSS delta (as reported by getrusage()) was "
            << ru.ru_maxrss - max_rss_before << " kB";
  LOG(INFO) << "Elapsed time: " << time_after - time_before << " ms";
}


TEST_F(SparseMerkleTreeTest, TestSetLeaf) {
  tree_.CurrentRoot();
  LOG(INFO) << "Tree@0:";
  LOG(INFO) << tree_.Dump();

  tree_.SetLeaf(PathFromString("one"), "one");
  tree_.CurrentRoot();
  LOG(INFO) << "Tree@1:";
  LOG(INFO) << tree_.Dump();

  tree_.SetLeaf(PathFromString("two"), "two");
  tree_.CurrentRoot();
  LOG(INFO) << "Tree@2:";
  LOG(INFO) << tree_.Dump();

  tree_.SetLeaf(PathFromString("three"), "three");
  tree_.CurrentRoot();
  LOG(INFO) << "Tree@3:";
  LOG(INFO) << tree_.Dump();
}


// TODO(alcutter): Lots and lots more tests.


}  // namespace


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
