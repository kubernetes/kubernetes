//go:build appengine
// +build appengine

// This file contains the safe implementations of otherwise unsafe-using code.

package xxhash

// Sum64String computes the 64-bit xxHash digest of s with a zero seed.
func Sum64String(s string) uint64 {
	return Sum64([]byte(s))
}

// WriteString adds more data to d. It always returns len(s), nil.
func (d *Digest) WriteString(s string) (n int, err error) {
	return d.Write([]byte(s))
}
