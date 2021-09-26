// +build !go1.9

package xxhash

// TODO(caleb): After Go 1.10 comes out, remove this fallback code.

func rol1(x uint64) uint64  { return (x << 1) | (x >> (64 - 1)) }
func rol7(x uint64) uint64  { return (x << 7) | (x >> (64 - 7)) }
func rol11(x uint64) uint64 { return (x << 11) | (x >> (64 - 11)) }
func rol12(x uint64) uint64 { return (x << 12) | (x >> (64 - 12)) }
func rol18(x uint64) uint64 { return (x << 18) | (x >> (64 - 18)) }
func rol23(x uint64) uint64 { return (x << 23) | (x >> (64 - 23)) }
func rol27(x uint64) uint64 { return (x << 27) | (x >> (64 - 27)) }
func rol31(x uint64) uint64 { return (x << 31) | (x >> (64 - 31)) }
