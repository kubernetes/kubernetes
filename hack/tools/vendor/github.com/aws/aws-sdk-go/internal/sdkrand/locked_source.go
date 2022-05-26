package sdkrand

import (
	"math/rand"
	"sync"
	"time"
)

// lockedSource is a thread-safe implementation of rand.Source
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

// SeededRand is a new RNG using a thread safe implementation of rand.Source
var SeededRand = rand.New(&lockedSource{src: rand.NewSource(time.Now().UnixNano())})
