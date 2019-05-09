// +build !go1.9

package swag

import (
	"sort"
	"sync"
)

// indexOfInitialisms is a thread-safe implementation of the sorted index of initialisms.
// Before go1.9, this may be implemented with a mutex on the map.
type indexOfInitialisms struct {
	getMutex *sync.Mutex
	index    map[string]bool
}

func newIndexOfInitialisms() *indexOfInitialisms {
	return &indexOfInitialisms{
		getMutex: new(sync.Mutex),
		index:    make(map[string]bool, 50),
	}
}

func (m *indexOfInitialisms) load(initial map[string]bool) *indexOfInitialisms {
	m.getMutex.Lock()
	defer m.getMutex.Unlock()
	for k, v := range initial {
		m.index[k] = v
	}
	return m
}

func (m *indexOfInitialisms) isInitialism(key string) bool {
	m.getMutex.Lock()
	defer m.getMutex.Unlock()
	_, ok := m.index[key]
	return ok
}

func (m *indexOfInitialisms) add(key string) *indexOfInitialisms {
	m.getMutex.Lock()
	defer m.getMutex.Unlock()
	m.index[key] = true
	return m
}

func (m *indexOfInitialisms) sorted() (result []string) {
	m.getMutex.Lock()
	defer m.getMutex.Unlock()
	for k := range m.index {
		result = append(result, k)
	}
	sort.Sort(sort.Reverse(byLength(result)))
	return
}
