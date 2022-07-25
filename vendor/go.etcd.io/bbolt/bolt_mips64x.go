// +build mips64 mips64le

package bbolt

// maxMapSize represents the largest mmap size supported by Bolt.
const maxMapSize = 0x8000000000 // 512GB

// maxAllocSize is the size used when creating array pointers.
const maxAllocSize = 0x7FFFFFFF
