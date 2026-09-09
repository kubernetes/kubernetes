//go:build ppc

package common

// MaxMapSize represents the largest mmap size supported by Bolt.
const MaxMapSize = 0x7FFFFFFF // 2GB

// MaxAllocSize is the size used when creating array pointers.
const MaxAllocSize = 0xFFFFFFF
