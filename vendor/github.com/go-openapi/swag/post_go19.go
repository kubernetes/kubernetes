// +build go1.9

package swag

import (
	"sort"
	"sync"
)

// indexOfInitialisms is a thread-safe implementation of the sorted index of initialisms.
// Since go1.9, this may be implemented with sync.Map.
type indexOfInitialisms struct {
	sortMutex *sync.Mutex
	index     *sync.Map
}

func newIndexOfInitialisms() *indexOfInitialisms {
	return &indexOfInitialisms{
		sortMutex: new(sync.Mutex),
		index:     new(sync.Map),
	}
}

func (m *indexOfInitialisms) load(initial map[string]bool) *indexOfInitialisms {
	m.sortMutex.Lock()
	defer m.sortMutex.Unlock()
	for k, v := range initial {
		m.index.Store(k, v)
	}
	return m
}

func (m *indexOfInitialisms) isInitialism(key string) bool {
	_, ok := m.index.Load(key)
	return ok
}

func (m *indexOfInitialisms) add(key string) *indexOfInitialisms {
	m.index.Store(key, true)
	return m
}

func (m *indexOfInitialisms) sorted() (result []string) {
	m.sortMutex.Lock()
	defer m.sortMutex.Unlock()
	m.index.Range(func(key, value interface{}) bool {
		k := key.(string)
		result = append(result, k)
		return true
	})
	sort.Sort(sort.Reverse(byLength(result)))
	return
}
