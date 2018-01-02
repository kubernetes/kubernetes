package pool

// Bytes is a pool of byte slices that can be re-used.  Slices in
// this pool will not be garbage collected when not in use.
type Bytes struct {
	pool chan []byte
}

// NewBytes returns a Bytes pool with capacity for max byte slices
// to be pool.
func NewBytes(max int) *Bytes {
	return &Bytes{
		pool: make(chan []byte, max),
	}
}

// Get returns a byte slice size with at least sz capacity. Items
// returned may not be in the zero state and should be reset by the
// caller.
func (p *Bytes) Get(sz int) []byte {
	var c []byte
	select {
	case c = <-p.pool:
	default:
		return make([]byte, sz)
	}

	if cap(c) < sz {
		return make([]byte, sz)
	}

	return c[:sz]
}

// Put returns a slice back to the pool.  If the pool is full, the byte
// slice is discarded.
func (p *Bytes) Put(c []byte) {
	select {
	case p.pool <- c:
	default:
	}
}
