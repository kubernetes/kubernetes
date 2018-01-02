package limiter

// Fixed is a simple channel based concurrency limiter.  It uses a fixed
// size channel to limit callers from proceeding until there is a value avalable
// in the channel.  If all are in-use, the caller blocks until one is freed.
type Fixed chan struct{}

func NewFixed(limit int) Fixed {
	return make(Fixed, limit)
}

func (t Fixed) Take() {
	t <- struct{}{}
}

func (t Fixed) Release() {
	<-t
}
