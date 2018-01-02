package sign

import (
	"bytes"
	"encoding/binary"
	"math/rand"
)

// A randomReader wraps a math/rand.Rand within an reader so that it can used
// as a predictable testing replacement for crypto/rand.Reader
type randomReader struct {
	b *bytes.Buffer
	r *rand.Rand
}

// newRandomReader returns a new instance of the random reader
func newRandomReader(r *rand.Rand) *randomReader {
	return &randomReader{b: &bytes.Buffer{}, r: r}
}

// Read will read random bytes from up to the length of b.
func (m *randomReader) Read(b []byte) (int, error) {
	for i := 0; i < len(b); {
		binary.Write(m.b, binary.LittleEndian, m.r.Int63())
		n, _ := m.b.Read(b[i:])
		i += n
	}

	return len(b), nil
}
