// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
	sort.Sort(sort.Reverse(byInitialism(result)))
	return
}
