package random

import (
	"math/rand"
	"sync"
	"time"
)

// copypaste from standard math/rand
type lockedSource struct {
	lk  sync.Mutex
	src rand.Source
}

func (r *lockedSource) Int63() (n int64) {
	r.lk.Lock()
	n = r.src.Int63()
	r.lk.Unlock()
	return
}

func (r *lockedSource) Seed(seed int64) {
	r.lk.Lock()
	r.src.Seed(seed)
	r.lk.Unlock()
}

// NewSource returns math/rand.Source safe for concurrent use and initialized
// with current unix-nano timestamp
func NewSource() rand.Source {
	return &lockedSource{
		src: rand.NewSource(time.Now().UnixNano()),
	}
}
