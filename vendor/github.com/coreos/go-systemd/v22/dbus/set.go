// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dbus

import (
	"sync"
)

type set struct {
	data map[string]bool
	mu   sync.Mutex
}

func (s *set) Add(value string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[value] = true
}

func (s *set) Remove(value string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data, value)
}

func (s *set) Contains(value string) (exists bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, exists = s.data[value]
	return
}

func (s *set) Length() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.data)
}

func (s *set) Values() (values []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for val := range s.data {
		values = append(values, val)
	}
	return
}

func newSet() *set {
	return &set{data: make(map[string]bool)}
}
