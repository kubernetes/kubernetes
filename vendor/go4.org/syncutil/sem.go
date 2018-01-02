package syncutil

import (
	"fmt"
	"log"
	"sync"
)

type debugT bool

var debug = debugT(false)

func (d debugT) Printf(format string, args ...interface{}) {
	if bool(d) {
		log.Printf(format, args...)
	}
}

// Sem implements a semaphore that can have multiple units acquired/released
// at a time.
type Sem struct {
	c         *sync.Cond // Protects size
	max, free int64
}

// NewSem creates a semaphore with max units available for acquisition.
func NewSem(max int64) *Sem {
	return &Sem{
		c:    sync.NewCond(new(sync.Mutex)),
		free: max,
		max:  max,
	}
}

// Acquire will deduct n units from the semaphore.  If the deduction would
// result in the available units falling below zero, the call will block until
// another go routine returns units via a call to Release.  If more units are
// requested than the semaphore is configured to hold, error will be non-nil.
func (s *Sem) Acquire(n int64) error {
	if n > s.max {
		return fmt.Errorf("sem: attempt to acquire more units than semaphore size %d > %d", n, s.max)
	}
	s.c.L.Lock()
	defer s.c.L.Unlock()
	for {
		debug.Printf("Acquire check max %d free %d, n %d", s.max, s.free, n)
		if s.free >= n {
			s.free -= n
			return nil
		}
		debug.Printf("Acquire Wait max %d free %d, n %d", s.max, s.free, n)
		s.c.Wait()
	}
}

// Release will return n units to the semaphore and notify any currently
// blocking Acquire calls.
func (s *Sem) Release(n int64) {
	s.c.L.Lock()
	defer s.c.L.Unlock()
	debug.Printf("Release max %d free %d, n %d", s.max, s.free, n)
	s.free += n
	s.c.Broadcast()
}
