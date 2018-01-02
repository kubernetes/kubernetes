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

func BenchmarkWriteAtBufferOrderedWrites(b *testing.B) {
	// test the performance of a WriteAtBuffer when written in an
	// ordered fashion. This is similar to the behavior of the
	// s3.Downloader, since downloads the first chunk of the file, then
	// the second, and so on.
	//
	// This test simulates a 150MB file being written in 30 ordered 5MB chunks.
	chunk := int64(5e6)
	max := chunk * 30
	// we'll write the same 5MB chunk every time
	tmp := make([]byte, chunk)
	for i := 0; i < b.N; i++ {
		buf := &WriteAtBuffer{}
		for i := int64(0); i < max; i += chunk {
			buf.WriteAt(tmp, i)
		}
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
