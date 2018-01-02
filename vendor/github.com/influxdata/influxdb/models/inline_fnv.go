package models // import "github.com/influxdata/influxdb/models"

// from stdlib hash/fnv/fnv.go
const (
	prime64  = 1099511628211
	offset64 = 14695981039346656037
)

// InlineFNV64a is an alloc-free port of the standard library's fnv64a.
type InlineFNV64a uint64

func NewInlineFNV64a() InlineFNV64a {
	return offset64
}

func (s *InlineFNV64a) Write(data []byte) (int, error) {
	hash := uint64(*s)
	for _, c := range data {
		hash ^= uint64(c)
		hash *= prime64
	}
	*s = InlineFNV64a(hash)
	return len(data), nil
}
func (s *InlineFNV64a) Sum64() uint64 {
	return uint64(*s)
}
