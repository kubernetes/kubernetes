// +build appengine

// This file contains the safe implementations of otherwise unsafe-using code.

package xxhash

// Sum64String computes the 64-bit xxHash digest of s.
func Sum64String(s string) uint64 {
	return Sum64([]byte(s))
}
