//go:build !(amd64 || arm64 || ppc64le || riscv64) || nounsafe || purego || appengine

package le

import (
	"encoding/binary"
)

// Load8 will load from b at index i.
func Load8[I Indexer](b []byte, i I) byte {
	return b[i]
}

// Load16 will load from b at index i.
func Load16[I Indexer](b []byte, i I) uint16 {
	return binary.LittleEndian.Uint16(b[i:])
}

// Load32 will load from b at index i.
func Load32[I Indexer](b []byte, i I) uint32 {
	return binary.LittleEndian.Uint32(b[i:])
}

// Load64 will load from b at index i.
func Load64[I Indexer](b []byte, i I) uint64 {
	return binary.LittleEndian.Uint64(b[i:])
}

// Store16 will store v at b.
func Store16(b []byte, v uint16) {
	binary.LittleEndian.PutUint16(b, v)
}

// Store32 will store v at b.
func Store32(b []byte, v uint32) {
	binary.LittleEndian.PutUint32(b, v)
}

// Store64 will store v at b.
func Store64(b []byte, v uint64) {
	binary.LittleEndian.PutUint64(b, v)
}
