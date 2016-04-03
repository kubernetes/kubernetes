package aws

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWriteAtBuffer(t *testing.T) {
	b := &WriteAtBuffer{}

	n, err := b.WriteAt([]byte{1}, 0)
	assert.NoError(t, err)
	assert.Equal(t, 1, n)

	n, err = b.WriteAt([]byte{1, 1, 1}, 5)
	assert.NoError(t, err)
	assert.Equal(t, 3, n)

	n, err = b.WriteAt([]byte{2}, 1)
	assert.NoError(t, err)
	assert.Equal(t, 1, n)

	n, err = b.WriteAt([]byte{3}, 2)
	assert.NoError(t, err)
	assert.Equal(t, 1, n)

	assert.Equal(t, []byte{1, 2, 3, 0, 0, 1, 1, 1}, b.Bytes())
}

func BenchmarkWriteAtBuffer(b *testing.B) {
	buf := &WriteAtBuffer{}
	r := rand.New(rand.NewSource(1))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		to := r.Intn(10) * 4096
		bs := make([]byte, to)
		buf.WriteAt(bs, r.Int63n(10)*4096)
	}
}

func BenchmarkWriteAtBufferParallel(b *testing.B) {
	buf := &WriteAtBuffer{}
	r := rand.New(rand.NewSource(1))

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			to := r.Intn(10) * 4096
			bs := make([]byte, to)
			buf.WriteAt(bs, r.Int63n(10)*4096)
		}
	})
}
