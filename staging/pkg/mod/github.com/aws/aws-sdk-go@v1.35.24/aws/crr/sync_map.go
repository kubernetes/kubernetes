// +build go1.9

package crr

import (
	"sync"
)

type syncMap sync.Map

func newSyncMap() syncMap {
	return syncMap{}
}

func (m *syncMap) Load(key interface{}) (interface{}, bool) {
	return (*sync.Map)(m).Load(key)
}

func (m *syncMap) Store(key interface{}, value interface{}) {
	(*sync.Map)(m).Store(key, value)
}

func (m *syncMap) Delete(key interface{}) {
	(*sync.Map)(m).Delete(key)
}

func (m *syncMap) Range(f func(interface{}, interface{}) bool) {
	(*sync.Map)(m).Range(f)
}
