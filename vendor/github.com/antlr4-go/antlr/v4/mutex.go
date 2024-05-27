//go:build !antlr.nomutex
// +build !antlr.nomutex

package antlr

import "sync"

// Mutex is a simple mutex implementation which just delegates to sync.Mutex, it
// is used to provide a mutex implementation for the antlr package, which users
// can turn off with the build tag -tags antlr.nomutex
type Mutex struct {
	mu sync.Mutex
}

func (m *Mutex) Lock() {
	m.mu.Lock()
}

func (m *Mutex) Unlock() {
	m.mu.Unlock()
}

type RWMutex struct {
	mu sync.RWMutex
}

func (m *RWMutex) Lock() {
	m.mu.Lock()
}

func (m *RWMutex) Unlock() {
	m.mu.Unlock()
}

func (m *RWMutex) RLock() {
	m.mu.RLock()
}

func (m *RWMutex) RUnlock() {
	m.mu.RUnlock()
}
