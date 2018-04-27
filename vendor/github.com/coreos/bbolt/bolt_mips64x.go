// +build mips64 mips64le

package bolt

// maxMapSize represents the largest mmap size supported by Bolt.
const maxMapSize = 0x8000000000 // 512GB

// maxAllocSize is the size used when creating array pointers.
const maxAllocSize = 0x7FFFFFFF

// Are unaligned load/stores broken on this arch?
var brokenUnaligned = false
