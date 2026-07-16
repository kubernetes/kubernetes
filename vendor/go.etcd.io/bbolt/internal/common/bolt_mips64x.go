//go:build mips64 || mips64le

package common

// MaxMapSize represents the largest mmap size supported by Bolt.
const MaxMapSize = 0x8000000000 // 512GB

// MaxAllocSize is the size used when creating array pointers.
const MaxAllocSize = 0x7FFFFFFF
