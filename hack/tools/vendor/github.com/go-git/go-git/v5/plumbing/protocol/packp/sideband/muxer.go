package sideband

import (
	"io"

	"github.com/go-git/go-git/v5/plumbing/format/pktline"
)

// Muxer multiplex the packfile along with the progress messages and the error
// information. The multiplex is perform using pktline format.
type Muxer struct {
	max int
	e   *pktline.Encoder
}

const chLen = 1

// NewMuxer returns a new Muxer for the given t that writes on w.
//
// If t is equal to `Sideband` the max pack size is set to MaxPackedSize, in any
// other value is given, max pack is set to MaxPackedSize64k, that is the
// maximum length of a line in pktline format.
func NewMuxer(t Type, w io.Writer) *Muxer {
	max := MaxPackedSize64k
	if t == Sideband {
		max = MaxPackedSize
	}

	return &Muxer{
		max: max - chLen,
		e:   pktline.NewEncoder(w),
	}
}

// Write writes p in the PackData channel
func (m *Muxer) Write(p []byte) (int, error) {
	return m.WriteChannel(PackData, p)
}

// WriteChannel writes p in the given channel. This method can be used with any
// channel, but is recommend use it only for the ProgressMessage and
// ErrorMessage channels and use Write for the PackData channel
func (m *Muxer) WriteChannel(t Channel, p []byte) (int, error) {
	wrote := 0
	size := len(p)
	for wrote < size {
		n, err := m.doWrite(t, p[wrote:])
		wrote += n

		if err != nil {
			return wrote, err
		}
	}

	return wrote, nil
}

func (m *Muxer) doWrite(ch Channel, p []byte) (int, error) {
	sz := len(p)
	if sz > m.max {
		sz = m.max
	}

	return sz, m.e.Encode(ch.WithPayload(p[:sz]))
}
