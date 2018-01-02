package merkletree

import (
	"bytes"
	"encoding/hex"
	"log"
	"reflect"
	"testing"
)

func TestCreateMerkleTree(t *testing.T) {
	tree := NewCPPMerkleTree()
	defer tree.DeletePeer()
	if tree == nil {
		t.Fatal("tree is nil")
	}
}

// Hex decodes |s| and returns the result.
// If the decode fails logs a fatal error
func mustDecode(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		log.Fatal(err)
	}
	return b
}

// Some test leaves
func testLeaves() [][]byte {
	return [][]byte{{},
		{0x00},
		{0x10},
		{0x20, 0x21},
		{0x30, 0x31},
		{0x40, 0x41, 0x42, 0x43},
		{0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57},
		{0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f}}
}

// Returns the correct root hash when |numLeaves| of the leaves returned by
// testLeaves() have been added to the tree in order.
// Logs a fatal error if |numLeaves| is too large.
func rootForTestLeaves(numLeaves int) []byte {
	switch numLeaves {
	case 0:
		return mustDecode("6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d")
	case 1:
		return mustDecode("fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125")
	case 2:
		return mustDecode("aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77")
	case 3:
		return mustDecode("d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7")
	case 4:
		return mustDecode("4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4")
	case 5:
		return mustDecode("76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef")
	case 6:
		return mustDecode("ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c")
	case 7:
		return mustDecode("5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328")
	default:
		log.Fatalf("Unexpected numLeaves %d", numLeaves)
	}
	return nil
}

func TestAddLeaf(t *testing.T) {
	m := NewCPPMerkleTree()
	defer m.DeletePeer()
	for index, a := range testLeaves() {
		i := m.AddLeaf(a)
		if i != uint64(index+1) {
			t.Fatalf("Got index %d, expected %d", i, index+1)
		}
		if m.LeafCount() != uint64(index+1) {
			t.Fatalf("LeafCount() %d, didn't match index+1 %d", m.LeafCount(), index+1)
		}
		r, err := m.CurrentRoot()
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Compare(r, rootForTestLeaves(index)) != 0 {
			t.Fatalf("CurrentRoot:\n%v\ndid not equal expected root:\n%v\n", hex.Dump(r), hex.Dump(rootForTestLeaves(index)))
		}
	}
}

func checkPath(t *testing.T, m *CPPMerkleTree, index uint64, expectedPath [][]byte) {
	path, err := m.PathToCurrentRoot(index)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(path, expectedPath) {
		t.Fatalf("Incorrect path returned for leaf@%d:\n%v\nexpected:\n%v", index, path, expectedPath)
	}
}

func TestPathToCurrentRoot(t *testing.T) {
	m := NewCPPMerkleTree()
	defer m.DeletePeer()
	for _, a := range testLeaves() {
		m.AddLeaf(a)
	}

	pathToOne := [][]byte{
		mustDecode("96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"),
		mustDecode("5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e"),
		mustDecode("6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4")}
	pathToSix := [][]byte{
		mustDecode("bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b"),
		mustDecode("ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0"),
		mustDecode("d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7")}

	checkPath(t, m, 1, pathToOne)
	checkPath(t, m, 6, pathToSix)
}

func checkConsistency(t *testing.T, m *CPPMerkleTree, from, to uint64, expectedProof [][]byte) {
	proof, err := m.SnapshotConsistency(from, to)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(proof, expectedProof) {
		t.Fatalf("Incorrect proof returned for consistency %d to %d:\n%v\nexpected:\n%v", from, to, proof, expectedProof)
	}
}

func TestSnapshotConsistency(t *testing.T) {
	m := NewCPPMerkleTree()
	defer m.DeletePeer()
	for _, a := range testLeaves() {
		m.AddLeaf(a)
	}

	oneToEight := [][]byte{
		mustDecode("96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"),
		mustDecode("5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e"),
		mustDecode("6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4")}
	sixToEight := [][]byte{
		mustDecode("0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a"),
		mustDecode("ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0"),
		mustDecode("d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7")}
	twoToFive := [][]byte{
		mustDecode("5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e"),
		mustDecode("bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b")}

	checkConsistency(t, m, 1, 8, oneToEight)
	checkConsistency(t, m, 6, 8, sixToEight)
	checkConsistency(t, m, 2, 5, twoToFive)
}

func TestAddLeafHash(t *testing.T) {
	hashValue := "0123456789abcdef0123456789abcdef"
	m := NewCPPMerkleTree()
	defer m.DeletePeer()
	index := m.AddLeafHash([]byte(hashValue))
	if index != 1 {
		t.Fatalf("Expected index of 1, got %d", index)
	}
	gotHash, err := m.LeafHash(index)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Compare([]byte(hashValue), gotHash) != 0 {
		t.Fatalf("Added leafhash:\n%v\nGot:\n%v", hex.Dump([]byte(hashValue)), hex.Dump(gotHash))
	}
}
