#include "merkletree/merkle_tree.h"

#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "_cgo_export.h"
#include "merkle_tree_go.h"

extern "C" {
// Some hollow functions to cast the void* types into what they really
// are, they're only really here to provide a little bit of type
// safety.  Hopefully these should all be optimized away into oblivion
// by the compiler.
static inline MerkleTree* MT(TREE tree) {
  assert(tree);
  return static_cast<MerkleTree*>(tree);
}
static inline Sha256Hasher* H(HASHER hasher) {
  assert(hasher);
  return static_cast<Sha256Hasher*>(hasher);
}
static inline GoSlice* BS(BYTE_SLICE slice) {
  assert(slice);
  return static_cast<GoSlice*>(slice);
}

HASHER NewSha256Hasher() {
  return new Sha256Hasher;
}

TREE NewMerkleTree(HASHER hasher) {
  return new MerkleTree(H(hasher));
}

void DeleteMerkleTree(TREE tree) {
  delete MT(tree);
}

size_t NodeSize(TREE tree) {
  return MT(tree)->NodeSize();
}

size_t LeafCount(TREE tree) {
  return MT(tree)->LeafCount();
}

bool LeafHash(TREE tree, BYTE_SLICE out, size_t leaf) {
  GoSlice* slice(BS(out));
  const MerkleTree* t(MT(tree));
  const size_t nodesize(t->NodeSize());
  if (slice->data == NULL || slice->cap < nodesize) {
    return false;
  }
  const std::string& hash = t->LeafHash(leaf);
  assert(nodesize == hash.size());
  memcpy(slice->data, hash.data(), nodesize);
  slice->len = nodesize;
  return true;
}

size_t LevelCount(TREE tree) {
  const MerkleTree* t(MT(tree));
  return t->LevelCount();
}

size_t AddLeaf(TREE tree, BYTE_SLICE leaf) {
  GoSlice* slice(BS(leaf));
  MerkleTree* t(MT(tree));
  return t->AddLeaf(std::string(static_cast<char*>(slice->data), slice->len));
}

size_t AddLeafHash(TREE tree, BYTE_SLICE hash) {
  GoSlice* slice(BS(hash));
  MerkleTree* t(MT(tree));
  return t->AddLeafHash(
      std::string(static_cast<char*>(slice->data), slice->len));
}

bool CurrentRoot(TREE tree, BYTE_SLICE out) {
  GoSlice* slice(BS(out));
  MerkleTree* t(MT(tree));
  const size_t nodesize(t->NodeSize());
  if (slice->data == NULL || slice->len != nodesize) {
    return false;
  }
  const std::string& hash = t->CurrentRoot();
  assert(nodesize == hash.size());
  memcpy(slice->data, hash.data(), nodesize);
  slice->len = nodesize;
  return true;
}

bool RootAtSnapshot(TREE tree, BYTE_SLICE out, size_t snapshot) {
  GoSlice* slice(BS(out));
  MerkleTree* t(MT(tree));
  const size_t nodesize(t->NodeSize());
  if (slice->data == NULL || slice->len != nodesize) {
    return false;
  }
  const std::string& hash = t->RootAtSnapshot(snapshot);
  assert(nodesize == hash.size());
  memcpy(slice->data, hash.data(), nodesize);
  slice->len = nodesize;
  return true;
}

// Copies the fixed-length entries from |path| into the GoSlice
// pointed to by |dst|, one after the other in the same order.
// |num_copied| is set to the number of entries copied.
bool CopyNodesToSlice(const std::vector<std::string>& path, GoSlice* dst,
                      size_t nodesize, size_t* num_copied) {
  assert(dst);
  assert(num_copied);
  if (dst->cap < path.size() * nodesize) {
    *num_copied = 0;
    return false;
  }
  char* e = static_cast<char*>(dst->data);
  for (int i = 0; i < path.size(); ++i) {
    assert(nodesize == path[i].size());
    memcpy(e, path[i].data(), nodesize);
    e += nodesize;
  }
  dst->len = path.size() * nodesize;
  *num_copied = path.size();
  return true;
}

bool PathToCurrentRoot(TREE tree, BYTE_SLICE out, size_t* num_entries,
                       size_t leaf) {
  MerkleTree* t(MT(tree));
  const std::vector<std::string> path = t->PathToCurrentRoot(leaf);
  return CopyNodesToSlice(path, BS(out), t->NodeSize(), num_entries);
}

bool PathToRootAtSnapshot(TREE tree, BYTE_SLICE out, size_t* num_entries,
                          size_t leaf, size_t snapshot) {
  MerkleTree* t(MT(tree));
  const std::vector<std::string> path =
      t->PathToRootAtSnapshot(leaf, snapshot);
  return CopyNodesToSlice(path, BS(out), t->NodeSize(), num_entries);
}

bool SnapshotConsistency(TREE tree, BYTE_SLICE out, size_t* num_entries,
                         size_t snapshot1, size_t snapshot2) {
  MerkleTree* t(MT(tree));
  const std::vector<std::string> path =
      t->SnapshotConsistency(snapshot1, snapshot2);
  return CopyNodesToSlice(path, BS(out), t->NodeSize(), num_entries);
}

}  // extern "C"
