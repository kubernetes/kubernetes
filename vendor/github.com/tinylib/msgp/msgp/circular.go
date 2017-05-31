package msgp

type timer interface {
	StartTimer()
	StopTimer()
}

// EndlessReader is an io.Reader
// that loops over the same data
// endlessly. It is used for benchmarking.
type EndlessReader struct {
	tb     timer
	data   []byte
	offset int
}

// NewEndlessReader returns a new endless reader
func NewEndlessReader(b []byte, tb timer) *EndlessReader {
	return &EndlessReader{tb: tb, data: b, offset: 0}
}

// Read implements io.Reader. In practice, it
// always returns (len(p), nil), although it
// fills the supplied slice while the benchmark
// timer is stopped.
func (c *EndlessReader) Read(p []byte) (int, error) {
	c.tb.StopTimer()
	var n int
	l := len(p)
	m := len(c.data)
	for n < l {
		nn := copy(p[n:], c.data[c.offset:])
		n += nn
		c.offset += nn
		c.offset %= m
	}
	c.tb.StartTimer()
	return n, nil
}
