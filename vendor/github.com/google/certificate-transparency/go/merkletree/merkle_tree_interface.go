package merkletree

// MerkleTreeInterface represents the common interface for basic MerkleTree functions.
type MerkleTreeInterface interface {
	// LeafCount returns the number of leaves in the tree
	LeafCount() uint64

	// LevelCount returns the number of levels in the tree
	LevelCount() uint64

	// AddLeaf adds the hash of |leaf| to the tree and returns the newly added
	// leaf index
	AddLeaf(leaf []byte) uint64

	// LeafHash returns the hash of the leaf at index |leaf| or a non-nil error.
	LeafHash(leaf uint64) ([]byte, error)

	// CurrentRoot returns the current root hash of the merkle tree.
	CurrentRoot() ([]byte, error)
}

// FullMerkleTreeInterface extends MerkleTreeInterface to the full range of
// operations that only a non-compact tree representation can implement.
type FullMerkleTreeInterface interface {
	MerkleTreeInterface

	// RootAtSnapshot returns the root hash at the tree size |snapshot|
	// which must be <= than the current tree size.
	RootAtSnapshot(snapshot uint64) ([]byte, error)

	// PathToCurrentRoot returns the Merkle path (or inclusion proof) from the
	// leaf hash at index |leaf| to the current root.
	PathToCurrentRoot(leaf uint64) ([]byte, error)

	// SnapshotConsistency returns a consistency proof between the two tree
	// sizes specified in |snapshot1| and |snapshot2|.
	SnapshotConsistency(snapshot1, snapshot2 uint64) ([]byte, error)
}
