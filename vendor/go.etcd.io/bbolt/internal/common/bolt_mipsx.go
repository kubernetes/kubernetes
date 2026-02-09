//go:build mips || mipsle

package common

// MaxMapSize represents the largest mmap size supported by Bolt.
const MaxMapSize = 0x40000000 // 1GB

// MaxAllocSize is the size used when creating array pointers.
const MaxAllocSize = 0xFFFFFFF
