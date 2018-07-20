// +build !go1.3

package v1

// Stub version of buffer_pool.go for Go 1.2, which doesn't have sync.Pool.

func Pool(b []byte) {}

func makeSlice(n int) []byte {
	return make([]byte, n)
}
