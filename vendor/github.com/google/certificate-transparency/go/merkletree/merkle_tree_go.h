#include <stdbool.h>
#include <sys/types.h>

#ifndef GO_MERKLETREE_MERKLE_TREE_H_
#define GO_MERKLETREE_MERKLE_TREE_H_

// These types & functions provide a trampoline to call the C++ MerkleTree
// implementation from within Go code.
//
// Generally we try to jump through hoops to not allocate memory from the C++
// side, but rather have Go allocate it inside its GC memory such that we don't
// have to worry about leaks.  Apart from the obvious benefit of doing it this
// way, it usually also means one less memcpy() too which is nice.

#ifdef __cplusplus
extern "C" {
#endif

// The _cgo_export.h file doesn't appear to exist when this header is pulled in
// to the .go file, because of this we can't use types like GoSlice here and so
// we end up with void* everywhere;  we'll at least typedef them so that the
// source is a _little_ more readable.
// Grumble grumble.
typedef void* HASHER;
typedef void* TREE;
typedef void* BYTE_SLICE;

// Allocators & deallocators:

// Creates a new Sha256Hasher
HASHER NewSha256Hasher();

// Creates a new MerkleTree passing in |hasher|.
// The MerkleTree takes ownership of |hasher|.
TREE NewMerkleTree(HASHER hasher);

// Deletes the passed in |tree|.
void DeleteMerkleTree(TREE tree);

// MerkleTree methods below.
// See the comments in ../../merkletree/merkle_tree.h for details

size_t NodeSize(TREE tree);
size_t LeafCount(TREE tree);
bool LeafHash(TREE tree, BYTE_SLICE out, size_t leaf);
size_t LevelCount(TREE tree);
size_t AddLeaf(TREE tree, BYTE_SLICE leaf);
size_t AddLeafHash(TREE tree, BYTE_SLICE hash);
bool CurrentRoot(TREE tree, BYTE_SLICE out);
bool RootAtSnapshot(TREE tree, BYTE_SLICE out, size_t snapshot);

// |out| must contain sufficent space to hold all of the path elements
// sequentially.
// |num_entries| is set to the number of actual elements stored in |out|.
bool PathToCurrentRoot(TREE tree, BYTE_SLICE out, size_t* num_entries,
                       size_t leaf);

// |out| must contain sufficent space to hold all of the path elements
// sequentially.
// |num_entries| is set to the number of actual elements stored in |out|.
bool PathToRootAtSnapshot(TREE tree, BYTE_SLICE out, size_t* num_entries,
                          size_t leaf, size_t snapshot);

// |out| must contain sufficent space to hold all of the path elements
// sequentially.
// |num_entries| is set to the number of actual elements stored in |out|.
bool SnapshotConsistency(TREE tree, BYTE_SLICE out, size_t* num_entries,
                         size_t snapshot1, size_t snapshot2);

#ifdef __cplusplus
}
#endif

#endif  // GO_MERKLETREE_MERKLE_TREE_H_
