package pool

// Generic is a pool of types that can be re-used.  Items in
// this pool will not be garbage collected when not in use.
type Generic struct {
	pool chan interface{}
	fn   func(sz int) interface{}
}

// NewGeneric returns a Generic pool with capacity for max items
// to be pool.
func NewGeneric(max int, fn func(sz int) interface{}) *Generic {
	return &Generic{
		pool: make(chan interface{}, max),
		fn:   fn,
	}
}

// Get returns a item from the pool or a new instance if the pool
// is empty.  Items returned may not be in the zero state and should
// be reset by the caller.
func (p *Generic) Get(sz int) interface{} {
	var c interface{}
	select {
	case c = <-p.pool:
	default:
		c = p.fn(sz)
	}

	return c
}

// Put returns an item back to the pool.  If the pool is full, the item
// is discarded.
func (p *Generic) Put(c interface{}) {
	select {
	case p.pool <- c:
	default:
	}
}
