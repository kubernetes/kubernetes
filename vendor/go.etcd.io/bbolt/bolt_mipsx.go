//go:build mips || mipsle

package bbolt

// maxMapSize represents the largest mmap size supported by Bolt.
const maxMapSize = 0x40000000 // 1GB

// maxAllocSize is the size used when creating array pointers.
const maxAllocSize = 0xFFFFFFF
