//go:build go1.6
// +build go1.6

package sdkrand

import "math/rand"

// Read provides the stub for math.Rand.Read method support for go version's
// 1.6 and greater.
func Read(r *rand.Rand, p []byte) (int, error) {
	return r.Read(p)
}
