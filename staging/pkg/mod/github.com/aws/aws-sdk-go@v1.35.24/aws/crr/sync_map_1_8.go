// +build !go1.9

package crr

import (
	"sync"
)

type syncMap struct {
	container map[interface{}]interface{}
	lock      sync.RWMutex
}

func newSyncMap() syncMap {
	return syncMap{
		container: map[interface{}]interface{}{},
	}
}

func (m *syncMap) Load(key interface{}) (interface{}, bool) {
	m.lock.RLock()
	defer m.lock.RUnlock()

	v, ok := m.container[key]
	return v, ok
}

func (m *syncMap) Store(key interface{}, value interface{}) {
	m.lock.Lock()
	defer m.lock.Unlock()

	m.container[key] = value
}

func (m *syncMap) Delete(key interface{}) {
	m.lock.Lock()
	defer m.lock.Unlock()

	delete(m.container, key)
}

func (m *syncMap) Range(f func(interface{}, interface{}) bool) {
	for k, v := range m.container {
		if !f(k, v) {
			return
		}
	}
}
