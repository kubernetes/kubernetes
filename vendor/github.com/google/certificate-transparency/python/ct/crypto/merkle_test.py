#!/usr/bin/env python

import unittest

from collections import namedtuple
import hashlib
import math

from ct.crypto import error
from ct.crypto import merkle


class TreeHasherTest(unittest.TestCase):
    sha256_empty_hash = ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495"
                         "991b7852b855")
    sha256_leaves = [
        ("",
         "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"),
        ("101112131415161718191a1b1c1d1e1f",
         "3bfb960453ebaebf33727da7a1f4db38acc051d381b6da20d6d4e88f0eabfd7a")
        ]
    sha256_nodes = [
        ("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
         "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
         "1a378704c17da31e2d05b6d121c2bb2c7d76f6ee6fa8f983e596c2d034963c57")]

    # array of bytestrings of the following literals in hex
    test_vector_leaves = ["".join(chr(int(n, 16)) for n in s.split()) for s in [
        "",
        "00",
        "10",
        "20 21",
        "30 31",
        "40 41 42 43",
        "50 51 52 53 54 55 56 57",
        "60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f",
    ]]

    test_vector_hashes = [
        "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
        "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
        "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
        "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
        "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
        "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
        "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
        "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
    ]

    def test_empty_hash(self):
        hasher = merkle.TreeHasher()
        self.assertEqual(hasher.hash_empty().encode("hex"),
                         TreeHasherTest.sha256_empty_hash)

    def test_hash_leaves(self):
        hasher = merkle.TreeHasher()
        for leaf, val in TreeHasherTest.sha256_leaves:
            self.assertEqual(hasher.hash_leaf(leaf.decode("hex")).encode("hex"),
                             val)

    def test_hash_children(self):
        hasher = merkle.TreeHasher()
        for left, right, val in  TreeHasherTest.sha256_nodes:
            self.assertEqual(hasher.hash_children(
                left.decode("hex"), right.decode("hex")).encode("hex"), val)

    def test_hash_full_invalid_index(self):
        hasher = merkle.TreeHasher()
        self.assertRaises(IndexError, hasher._hash_full, "abcd", -5, -1)
        self.assertRaises(IndexError, hasher._hash_full, "abcd", -1, 1)
        self.assertRaises(IndexError, hasher._hash_full, "abcd", 1, 5)
        self.assertRaises(IndexError, hasher._hash_full, "abcd", 2, 1)

    def test_hash_full_empty(self):
        hasher = merkle.TreeHasher()
        for i in xrange(0, 5):
            self.assertEqual(hasher._hash_full("abcd", i, i)[0].encode("hex"),
                              TreeHasherTest.sha256_empty_hash)

    def test_hash_full_tree(self):
        hasher = merkle.TreeHasher()
        self.assertEqual(hasher.hash_full_tree([]), hasher.hash_empty())
        l = iter(hasher.hash_leaf(c) for c in "abcde").next
        h = hasher.hash_children
        root_hash = h(h(h(l(), l()), h(l(), l())), l())
        self.assertEqual(hasher.hash_full_tree("abcde"), root_hash)

    def test_hash_full_tree_test_vector(self):
        hasher = merkle.TreeHasher()
        for i in xrange(len(TreeHasherTest.test_vector_leaves)):
            test_vector = TreeHasherTest.test_vector_leaves[:i+1]
            expected_hash = TreeHasherTest.test_vector_hashes[i].decode("hex")
            self.assertEqual(hasher.hash_full_tree(test_vector), expected_hash)


class HexTreeHasher(merkle.TreeHasher):
    def __init__(self, hashfunc=hashlib.sha256):
        self.hasher = merkle.TreeHasher(hashfunc)

    def hash_empty(self):
        return self.hasher.hash_empty().encode("hex")

    def hash_leaf(self, data):
        return self.hasher.hash_leaf(data.decode("hex")).encode("hex")

    def hash_children(self, left, right):
        return self.hasher.hash_children(left.decode("hex"),
                                         right.decode("hex")).encode("hex")


class CompactMerkleTreeTest(unittest.TestCase):

    def setUp(self):
        self.tree = merkle.CompactMerkleTree(HexTreeHasher())

    def test_extend_from_empty(self):
        for i in xrange(len(TreeHasherTest.test_vector_leaves)):
            test_vector = TreeHasherTest.test_vector_leaves[:i+1]
            expected_hash = TreeHasherTest.test_vector_hashes[i]
            self.tree = merkle.CompactMerkleTree()
            self.tree.extend(test_vector)
            self.assertEqual(self.tree.root_hash().encode("hex"), expected_hash)

    def test_push_subtree_1(self):
        for i in xrange(len(TreeHasherTest.test_vector_leaves)):
            test_vector = TreeHasherTest.test_vector_leaves[:i+1]
            self.tree = merkle.CompactMerkleTree()
            self.tree.extend(test_vector)
            self.tree._push_subtree(["test leaf"])
            self.assertEqual(len(self.tree), len(test_vector) + 1)

    def test_extend_from_partial(self):
        z = len(TreeHasherTest.test_vector_leaves)
        for i in xrange(z):
            self.tree = merkle.CompactMerkleTree()
            # add up to i
            test_vector = TreeHasherTest.test_vector_leaves[:i+1]
            expected_hash = TreeHasherTest.test_vector_hashes[i]
            self.tree.extend(test_vector)
            self.assertEqual(self.tree.root_hash().encode("hex"), expected_hash)
            # add up to z
            test_vector = TreeHasherTest.test_vector_leaves[i+1:]
            expected_hash = TreeHasherTest.test_vector_hashes[z-1]
            self.tree.extend(test_vector)
            self.assertEqual(self.tree.root_hash().encode("hex"), expected_hash)


class MerkleVerifierTest(unittest.TestCase):
    # (old_tree_size, new_tree_size, old_root, new_root, proof)
    # Test vectors lifted from the C++ branch.
    sha256_proofs = [
        (1, 1,
         "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
         "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
         []),
        (1, 8,
         "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
         "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
         ["96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7",
          "5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e",
          "6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4"]),
        (6, 8,
         "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
         "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
         ["0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a",
          "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0",
          "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7"]),
        (2, 5,
         "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
         "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
         ["5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e",
          "bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b"])
        ]

    # Data for leaf inclusion proof test
    sha256_audit_path = [
        "1a208aeebcd1b39fe2de247ee8db9454e1e93a312d206b87f6ca9cc6ec6f1ddd",
        "0a1b78b383f580856f433c01a5741e160d451c185910027f6cc9f828687a40c4",
        "3d1745789bc63f2da15850de1c12a5bf46ed81e1cc90f086148b1662e79aab3d",
        "9095b61e14d8990acf390905621e62b1714fb8e399fbb71de5510e0aef45affe",
        "0a332b91b8fab564e6afd1dd452449e04619b18accc0ff9aa8393cd4928451f2",
        "2336f0181d264aed6d8f3a6507ca14a8d3b3c3a23791ac263e845d208c1ee330",
        "b4ce56e300590500360c146c6452edbede25d4ed83919278749ee5dbe178e048",
        "933f6ddc848ea562e4f9c5cfb5f176941301dad0c6fdb9d1fbbe34fac1be6966",
        "b95a6222958a86f74c030be27c44f57dbe313e5e7c7f4ffb98bcbd3a03bb52f2",
        "daeeb3ce5923defd0faeb8e0c210b753b85b809445d7d3d3cd537a9aabaa9c45",
        "7fadd0a13e9138a2aa6c3fdec4e2275af233b94812784f66bcca9aa8e989f2bc",
        "1864e6ba3e32878610546539734fb5eeae2529991f130c575c73a7e25a2a7c56",
        "12842d1202b1dc6828a17ab253c02e7ce9409b5192430feba44189f39cc02d66",
        "29af64b16fa3053c13d02ac63aa75b23aa468506e44c3a2315edc85d2dc22b11",
        "b527b99934a0bd9edd154e449b0502e2c499bba783f3bc3dfe23364b6b532009",
        "4584db8ae8e351ace08e01f306378a92bfd43611714814f3d834a2842d69faa8",
        "86a9a41573b0d6e4292f01e93243d6cc65b30f06606fc6fa57390e7e90ed580f",
        "a88b98fbe84d4c6aae8db9d1605dfac059d9f03fe0fcb0d5dff1295dacba09e6",
        "06326dc617a6d1f7021dc536026dbfd5fffc6f7c5531d48ef6ccd1ed1569f2a1",
        "f41fe8fdc3a2e4e8345e30216e7ebecffee26ff266eeced208a6c2a3cf08f960",
        "40cf5bde8abb76983f3e98ba97aa36240402975674e120f234b3448911090f8d",
        "b3222dc8658538079883d980d7fdc2bef9285344ea34338968f736b04aeb387a"]

    raw_hex_leaf = (
        "00000000013de9d2b29b000000055b308205573082043fa00302010202072b777b56df"
        "7bc5300d06092a864886f70d01010505003081ca310b30090603550406130255533110"
        "300e060355040813074172697a6f6e61311330110603550407130a53636f7474736461"
        "6c65311a3018060355040a1311476f44616464792e636f6d2c20496e632e3133303106"
        "0355040b132a687474703a2f2f6365727469666963617465732e676f64616464792e63"
        "6f6d2f7265706f7369746f72793130302e06035504031327476f204461646479205365"
        "637572652043657274696669636174696f6e20417574686f726974793111300f060355"
        "040513083037393639323837301e170d3133303131343038353035305a170d31353031"
        "31343038353035305a305331163014060355040a130d7777772e69646e65742e6e6574"
        "3121301f060355040b1318446f6d61696e20436f6e74726f6c2056616c696461746564"
        "311630140603550403130d7777772e69646e65742e6e657430820122300d06092a8648"
        "86f70d01010105000382010f003082010a0282010100d4e4a4b1bbc981c9b8166f0737"
        "c113000aa5370b21ad86a831a379de929db258f056ba0681c50211552b249a02ec00c5"
        "37e014805a5b5f4d09c84fdcdfc49310f4a9f9004245d119ce5461bc5c42fd99694b88"
        "388e035e333ac77a24762d2a97ea15622459cc4adcd37474a11c7cff6239f810120f85"
        "e014d2066a3592be604b310055e84a74c91c6f401cb7f78bdb45636fb0b1516b04c5ee"
        "7b3fa1507865ff885d2ace21cbb28fdaa464efaa1d5faab1c65e4c46d2139175448f54"
        "b5da5aea956719de836ac69cd3a74ca049557cee96f5e09e07ba7e7b4ebf9bf167f4c3"
        "bf8039a4cab4bec068c899e997bca58672bd7686b5c85ea24841e48c46f76830390203"
        "010001a38201b6308201b2300f0603551d130101ff04053003010100301d0603551d25"
        "0416301406082b0601050507030106082b06010505070302300e0603551d0f0101ff04"
        "04030205a030330603551d1f042c302a3028a026a0248622687474703a2f2f63726c2e"
        "676f64616464792e636f6d2f676473312d38332e63726c30530603551d20044c304a30"
        "48060b6086480186fd6d010717013039303706082b06010505070201162b687474703a"
        "2f2f6365727469666963617465732e676f64616464792e636f6d2f7265706f7369746f"
        "72792f30818006082b0601050507010104743072302406082b06010505073001861868"
        "7474703a2f2f6f6373702e676f64616464792e636f6d2f304a06082b06010505073002"
        "863e687474703a2f2f6365727469666963617465732e676f64616464792e636f6d2f72"
        "65706f7369746f72792f67645f696e7465726d6564696174652e637274301f0603551d"
        "23041830168014fdac6132936c45d6e2ee855f9abae7769968cce730230603551d1104"
        "1c301a820d7777772e69646e65742e6e6574820969646e65742e6e6574301d0603551d"
        "0e041604144d3ae8a87ddcf046764021b87e7d8d39ddd18ea0300d06092a864886f70d"
        "01010505000382010100ad651b199f340f043732a71178c0af48e22877b9e5d99a70f5"
        "d78537c31d6516e19669aa6bfdb8b2cc7a145ba7d77b35101f9519e03b58e692732314"
        "1383c3ab45dc219bd5a584a2b6333b6e1bbef5f76e89b3c187ef1d3b853b4910e895a4"
        "57dbe7627e759f56c8484c30b22a74fb00f7b1d7c41533a1fd176cd2a2b06076acd7ca"
        "ddc6ca6d0c2a815f9eb3ef0d03d27e7eebd7824c78fdb51679c03278cfbb2d85ae65a4"
        "7485cb733fc1c7407834f7471ababd68f140983817c6f388b2f2e2bfe9e26608f9924f"
        "16473462d136427d1f2801e4b870b078c20ec4ba21e22ab32a00b76522d523825bcabb"
        "8c7b6142d624be8d2af69ecc36fb5689572a0f59c00000")

    leaf_hash = (
        "7a395c866d5ecdb0cccb623e011dbc392cd348d1d1d72776174e127a24b09c78")
    leaf_index = 848049
    tree_size = 3630887
    expected_root_hash = (
        "78316a05c9bcf14a3a4548f5b854a9adfcd46a4c034401b3ce7eb7ac2f1d0ecb")


    def setUp(self):
        self.verifier = merkle.MerkleVerifier(HexTreeHasher())
        self.STH = namedtuple("STH", ["sha256_root_hash", "tree_size"])
        self.ones = "11" * 32
        self.zeros = "00" * 32

    def test_verify_tree_consistency(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        for test_vector in  MerkleVerifierTest.sha256_proofs:
            self.assertTrue(verifier.verify_tree_consistency(*test_vector))

    def test_verify_tree_consistency_always_accepts_empty_tree(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        # Give some bogus proof too; it should be ignored.
        self.assertTrue(verifier.verify_tree_consistency(
            0, 1,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            ["6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"]
            ))

    def test_verify_tree_consistency_for_equal_tree_sizes(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        # Equal tree sizes and hashes, and a bogus proof that should be ignored.
        self.assertTrue(verifier.verify_tree_consistency(
            3, 3,
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            ["6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"]
            ))

        # Equal tree sizes but different hashes.
        self.assertRaises(
            error.ConsistencyError, verifier.verify_tree_consistency, 3, 3,
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01e",
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            [])

    def test_verify_tree_consistency_newer_tree_is_smaller(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        self.assertRaises(
            ValueError, verifier.verify_tree_consistency, 5, 2,
            "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
            "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
            ["5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e",
             "bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b"]
            )

    def test_verify_tree_consistency_proof_too_short(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        self.assertRaises(
            error.ProofError, verifier.verify_tree_consistency, 6, 8,
            "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
            "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
            ["0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a",
             "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0"]
            )

    def test_verify_tree_consistency_bad_second_hash(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        # A bit has been flipped in the second hash.
        self.assertRaises(
            error.ProofError, verifier.verify_tree_consistency, 6, 8,
            "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
            "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604329",
            ["0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a",
             "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0",
             "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7"]
            )

    def test_verify_tree_consistency_both_hashes_bad(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        # A bit has been flipped in both hashes.
        self.assertRaises(
            error.ProofError, verifier.verify_tree_consistency, 6, 8,
            "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ee",
            "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604329",
            ["0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a",
             "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0",
             "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7"]
            )

    def test_verify_tree_consistency_bad_first_hash(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        # A bit has been flipped in the first hash.
        self.assertRaises(
            error.ConsistencyError, verifier.verify_tree_consistency, 6, 8,
            "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ee",
            "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
            ["0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a",
             "ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0",
             "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7"]
            )

    def test_calculate_root_hash_good_proof(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        self.assertEqual(
            verifier._calculate_root_hash_from_audit_path(
                self.leaf_hash, self.leaf_index, self.sha256_audit_path[:],
                self.tree_size),
            self.expected_root_hash)

    def test_calculate_root_too_short_proof(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        leaf_index = self.leaf_index + int(
            math.pow(2, len(self.sha256_audit_path) + 1))
        self.assertRaises(
            error.ProofError,
            verifier._calculate_root_hash_from_audit_path,
            self.leaf_hash, leaf_index, self.sha256_audit_path[:],
            self.tree_size)

    def test_verify_leaf_inclusion_good_proof(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        sth = self.STH(self.expected_root_hash, self.tree_size)
        self.assertTrue(
            verifier.verify_leaf_inclusion(
                self.raw_hex_leaf, self.leaf_index, self.sha256_audit_path,
                sth))

    def test_verify_leaf_inclusion_bad_proof(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        # Expect this test to fail by providing an incorrect root hash.
        sth = self.STH(self.zeros, self.tree_size)
        self.assertRaises(
            error.ProofError, verifier.verify_leaf_inclusion,
            self.raw_hex_leaf, self.leaf_index, self.sha256_audit_path, sth)

    def test_verify_leaf_inclusion_incorrect_length_proof(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        sth = self.STH(self.zeros, 4)
        # Too long a proof
        self.assertRaises(
            error.ProofError, verifier.verify_leaf_inclusion,
            self.ones, 0, [self.zeros, self.zeros, self.zeros], sth)
        # Too short a proof
        self.assertRaises(
            error.ProofError, verifier.verify_leaf_inclusion,
            self.ones, 0, [self.zeros], sth)

    def test_verify_leaf_inclusion_single_node_in_tree(self):
        # If there is only one entry in the tree, the tree root hash should be
        # equal to the leaf hash.
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        sth = self.STH(self.leaf_hash, 1)
        self.assertTrue(
            verifier.verify_leaf_inclusion(self.raw_hex_leaf, 0, [], sth))

    def test_verify_leaf_inclusion_rightmost_node_in_tree(self):
        # Show that verify_leaf_inclusion works when required to check a proof
        # for the right-most node: In a tree of 8 nodes, ask for inclusion
        # proof check for leaf 7.
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        hh = HexTreeHasher()
        h_s1 = hh.hash_leaf(self.ones)
        h_c3 = hh.hash_children(self.zeros, h_s1)
        h_c2 = hh.hash_children(self.zeros, h_c3)
        h_root = hh.hash_children(self.zeros, h_c2)
        sth = self.STH(h_root, 8)
        self.assertTrue(
            verifier.verify_leaf_inclusion(
                self.ones, 7, [self.zeros, self.zeros, self.zeros], sth))

    def test_verify_leaf_inclusion_rightmost_node_in_unbalanced_odd_tree(
        self):
        # Show that verify_leaf_inclusion works when required to check a proof
        # for the right-most, even-indexed node: In a tree of 5 nodes, ask for
        # inclusion proof check for leaf 4 (the 5th).
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        hh = HexTreeHasher()
        h_s1 = hh.hash_leaf(self.ones)
        h_root = hh.hash_children(self.zeros, h_s1)
        sth = self.STH(h_root, 5)
        self.assertTrue(
            verifier.verify_leaf_inclusion(self.ones, 4, [self.zeros, ], sth))

    def test_verify_leaf_inclusion_rightmost_node_in_unbalanced_tree_odd_node(
        self):
        # Show that verify_leaf_inclusion works when required to check a proof
        # for the right-most, odd-indexed node: In a tree of 6 nodes, ask for
        # inclusion proof check for leaf 5 (the 6th).
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        hh = HexTreeHasher()
        h_s1 = hh.hash_leaf(self.ones)
        h_l2 = hh.hash_children(self.zeros, h_s1)
        h_root = hh.hash_children(self.zeros, h_l2)
        sth = self.STH(h_root, 6)
        self.assertTrue(
            verifier.verify_leaf_inclusion(
                self.ones, 5, [self.zeros, self.zeros], sth))

    def test_verify_leaf_inclusion_rightmost_node_in_unbalanced_even_tree(
        self):
        # Show that verify_leaf_inclusion works when required to check a proof
        # for the right-most, odd-indexed node: In a tree of 6 nodes, ask for
        # inclusion proof check for leaf 4 (the 5th).
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        hh = HexTreeHasher()
        h_s1 = hh.hash_leaf(self.ones)
        h_l2 = hh.hash_children(h_s1, self.zeros)
        h_root = hh.hash_children(self.zeros, h_l2)
        sth = self.STH(h_root, 6)
        self.assertTrue(
            verifier.verify_leaf_inclusion(
                self.ones, 4, [self.zeros, self.zeros], sth))

    def test_verify_leaf_inclusion_throws_on_bad_indices(self):
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        sth = self.STH("", 6)
        self.assertRaises(ValueError,
            verifier.verify_leaf_inclusion, "", -3, [], sth)
        negative_sth = self.STH("", -3)
        self.assertRaises(ValueError,
            verifier.verify_leaf_inclusion, "", 3, [], negative_sth)

    def test_verify_leaf_inclusion_all_nodes_all_tree_sizes_up_to_4(self):
        leaves = ["aa", "bb", "cc", "dd"]
        hh = HexTreeHasher()
        leaf_hashes = [hh.hash_leaf(l) for l in leaves]
        hc = hh.hash_children
        proofs_per_tree_size = {
            1: [[] ],
            2: [[leaf_hashes[1]], [leaf_hashes[0]]],
            3: [[leaf_hashes[1], leaf_hashes[2]], # leaf 0
                [leaf_hashes[0], leaf_hashes[2]], # leaf 1
                [hc(leaf_hashes[0], leaf_hashes[1])]], # leaf 2
            4: [[leaf_hashes[1], hc(leaf_hashes[2], leaf_hashes[3])], # leaf 0
                [leaf_hashes[0], hc(leaf_hashes[2], leaf_hashes[3])], # leaf 1
                [leaf_hashes[3], hc(leaf_hashes[0], leaf_hashes[1])], # leaf 2
                [leaf_hashes[2], hc(leaf_hashes[0], leaf_hashes[1])], # leaf 3
                ]
            }
        tree = merkle.CompactMerkleTree(hasher=HexTreeHasher())
        verifier = merkle.MerkleVerifier(HexTreeHasher())
        # Increase the tree by one leaf each time
        for i in range(4):
            tree.append(leaves[i])
            tree_size = i + 1
            # ... and check inclusion proof validates for each node
            # of the tree
            for j in range(tree_size):
              proof = proofs_per_tree_size[tree_size][j]
              sth = self.STH(tree.root_hash(), tree_size)
              self.assertTrue(
                  verifier.verify_leaf_inclusion(
                      leaves[j], j, proof, sth))


if __name__ == "__main__":
    unittest.main()
