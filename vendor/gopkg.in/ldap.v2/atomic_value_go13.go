// +build !go1.4

package ldap

import (
	"sync"
)

// This is a helper type that emulates the use of the "sync/atomic.Value"
// struct that's available in Go 1.4 and up.
type atomicValue struct {
	value interface{}
	lock  sync.RWMutex
}

func (av *atomicValue) Store(val interface{}) {
	av.lock.Lock()
	av.value = val
	av.lock.Unlock()
}

func (av *atomicValue) Load() interface{} {
	av.lock.RLock()
	ret := av.value
	av.lock.RUnlock()

	return ret
}
