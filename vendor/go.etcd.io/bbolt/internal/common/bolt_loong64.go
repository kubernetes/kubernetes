//go:build loong64

package common

// MaxMapSize represents the largest mmap size supported by Bolt.
const MaxMapSize = 0xFFFFFFFFFFFF // 256TB

// MaxAllocSize is the size used when creating array pointers.
const MaxAllocSize = 0x7FFFFFFF
