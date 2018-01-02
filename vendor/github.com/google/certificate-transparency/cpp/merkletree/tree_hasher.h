#ifndef TREEHASHER_H
#define TREEHASHER_H

#include <stddef.h>
#include <memory>
#include <mutex>
#include <string>

#include "base/macros.h"
#include "merkletree/serial_hasher.h"

class TreeHasher {
 public:
  // Takes ownership of the SerialHasher.
  TreeHasher(SerialHasher* hasher);

  size_t DigestSize() const {
    return hasher_->DigestSize();
  }

  const std::string& HashEmpty() const {
    return empty_hash_;
  }

  std::string HashLeaf(const std::string& data) const;

  // Accepts arbitrary strings as children. When hashing digests, it
  // is the responsibility of the caller to ensure the inputs are of
  // correct size.
  std::string HashChildren(const std::string& left_child,
                           const std::string& right_child) const;

 private:
  mutable std::mutex lock_;
  const std::unique_ptr<SerialHasher> hasher_;
  // The pre-computed hash of an empty tree.
  const std::string empty_hash_;

  DISALLOW_COPY_AND_ASSIGN(TreeHasher);
};
#endif
