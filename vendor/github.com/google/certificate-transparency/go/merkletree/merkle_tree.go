package merkletree

/*
#cgo LDFLAGS: -lcrypto
#cgo CPPFLAGS: -I../../cpp
#cgo CXXFLAGS: -std=c++11
#include "merkle_tree_go.h"
*/
import "C"
import (
	"errors"
	"fmt"
)

// CPPMerkleTree provides an interface to the C++ CT MerkleTree library.
// See the go/README file for details on how to build this.
type CPPMerkleTree struct {
	FullMerkleTreeInterface

	// The C++ MerkleTree handle
	peer C.TREE

	// nodeSize contains the size in bytes of the nodes in the MerkleTree
	// referenced by |peer|.
	nodeSize C.size_t
}

func (m *CPPMerkleTree) LeafCount() uint64 {
	return uint64(C.LeafCount(m.peer))
}

func (m *CPPMerkleTree) LevelCount() uint64 {
	return uint64(C.LevelCount(m.peer))
}

func (m *CPPMerkleTree) AddLeaf(leaf []byte) uint64 {
	return uint64(C.AddLeaf(m.peer, C.BYTE_SLICE(&leaf)))
}

func (m *CPPMerkleTree) AddLeafHash(hash []byte) uint64 {
	return uint64(C.AddLeafHash(m.peer, C.BYTE_SLICE(&hash)))
}

func (m *CPPMerkleTree) LeafHash(leaf uint64) ([]byte, error) {
	hash := make([]byte, m.nodeSize)
	success := C.LeafHash(m.peer, C.BYTE_SLICE(&hash), C.size_t(leaf))
	if !success {
		return nil, fmt.Errorf("failed to get leafhash of leaf %d", leaf)
	}
	return hash, nil
}

func (m *CPPMerkleTree) CurrentRoot() ([]byte, error) {
	hash := make([]byte, m.nodeSize)
	success := C.CurrentRoot(m.peer, C.BYTE_SLICE(&hash))
	if !success {
		return nil, errors.New("failed to get current root")
	}
	return hash, nil
}

func (m *CPPMerkleTree) RootAtSnapshot(snapshot uint64) ([]byte, error) {
	hash := make([]byte, m.nodeSize)
	success := C.RootAtSnapshot(m.peer, C.BYTE_SLICE(&hash), C.size_t(snapshot))
	if !success {
		return nil, fmt.Errorf("failed to get root at snapshot %d", snapshot)
	}
	return hash, nil
}

func splitSlice(slice []byte, chunkSize int) ([][]byte, error) {
	if len(slice)%chunkSize != 0 {
		return nil, fmt.Errorf("slice len %d is not a multiple of chunkSize %d", len(slice), chunkSize)
	}
	numEntries := len(slice) / chunkSize
	ret := make([][]byte, numEntries)
	for i := 0; i < numEntries; i++ {
		start := i * chunkSize
		end := start + chunkSize
		ret[i] = slice[start:end]
	}
	return ret, nil
}

func (m *CPPMerkleTree) PathToCurrentRoot(leaf uint64) ([][]byte, error) {
	var numEntries C.size_t
	entryBuffer := make([]byte, C.size_t(m.LevelCount())*m.nodeSize)
	success := C.PathToCurrentRoot(m.peer, C.BYTE_SLICE(&entryBuffer), &numEntries, C.size_t(leaf))
	if !success {
		return nil, fmt.Errorf("failed to get path to current root from leaf %d", leaf)
	}
	return splitSlice(entryBuffer, int(m.nodeSize))
}

func (m *CPPMerkleTree) PathToRootAtSnapshot(leaf, snapshot uint64) ([][]byte, error) {
	var num_entries C.size_t
	entryBuffer := make([]byte, C.size_t(m.LevelCount())*m.nodeSize)
	success := C.PathToRootAtSnapshot(m.peer, C.BYTE_SLICE(&entryBuffer), &num_entries, C.size_t(leaf), C.size_t(snapshot))
	if !success {
		return nil, fmt.Errorf("failed to get path to root at snapshot %d from leaf %d", snapshot, leaf)
	}
	return splitSlice(entryBuffer, int(m.nodeSize))
}

func (m *CPPMerkleTree) SnapshotConsistency(snapshot1, snapshot2 uint64) ([][]byte, error) {
	var num_entries C.size_t
	entryBuffer := make([]byte, C.size_t(m.LevelCount())*m.nodeSize)
	success := C.SnapshotConsistency(m.peer, C.BYTE_SLICE(&entryBuffer), &num_entries, C.size_t(snapshot1), C.size_t(snapshot2))
	if !success {
		return nil, fmt.Errorf("failed to get path to snapshot consistency from %d to %d", snapshot1, snapshot2)
	}
	return splitSlice(entryBuffer, int(m.nodeSize))
}

// NewCPPMerkleTree returns a new wrapped C++ MerkleTree, using the
// Sha256Hasher.
// It is the caller's responsibility to call DeletePeer() when finished with
// the tree to deallocate its resources.
func NewCPPMerkleTree() *CPPMerkleTree {
	m := &CPPMerkleTree{
		peer: C.NewMerkleTree(C.NewSha256Hasher()),
	}
	m.nodeSize = C.size_t(C.NodeSize(m.peer))
	return m
}

// DeletePeer deallocates the memory used by the C++ MerkleTree peer.
func (m *CPPMerkleTree) DeletePeer() {
	C.DeleteMerkleTree(m.peer)
	m.peer = nil
}
