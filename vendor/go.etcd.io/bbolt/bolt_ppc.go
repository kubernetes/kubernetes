// +build ppc

package bbolt

// maxMapSize represents the largest mmap size supported by Bolt.
const maxMapSize = 0x7FFFFFFF // 2GB

// maxAllocSize is the size used when creating array pointers.
const maxAllocSize = 0xFFFFFFF
