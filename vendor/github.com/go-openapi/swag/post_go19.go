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

//go:build go1.9
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
	sort.Sort(sort.Reverse(byInitialism(result)))
	return
}
