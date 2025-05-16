//go:build ppc64le

package bbolt

// maxMapSize represents the largest mmap size supported by Bolt.
const maxMapSize = 0xFFFFFFFFFFFF // 256TB

// maxAllocSize is the size used when creating array pointers.
const maxAllocSize = 0x7FFFFFFF
