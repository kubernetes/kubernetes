package utils

import "sync/atomic"

// An AtomicBool is an atomic bool
type AtomicBool struct {
	v int32
}

// Set sets the value
func (a *AtomicBool) Set(value bool) {
	var n int32
	if value {
		n = 1
	}
	atomic.StoreInt32(&a.v, n)
}

// Get gets the value
func (a *AtomicBool) Get() bool {
	return atomic.LoadInt32(&a.v) != 0
}
