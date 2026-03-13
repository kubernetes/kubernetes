/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package pullmanager

import (
	"hash/fnv"
	"sync"
)

// StripedLockSet allows context locking based on string keys, where each key
// is mapped to a an index in a size-limited slice of locks.
type StripedLockSet struct {
	locks []sync.Mutex
	size  int32
}

// NewStripedLockSet creates a StripedLockSet with `size` number of locks to be
// used for locking context based on string keys.
// The size will be normalized to stay in the <1, 31> interval.
func NewStripedLockSet(size int32) *StripedLockSet {
	size = max(size, 1) // make sure we're at least at size 1

	return &StripedLockSet{
		locks: make([]sync.Mutex, min(31, size)),
		size:  size,
	}
}

func (s *StripedLockSet) Lock(key string) {
	s.locks[keyToID(key, s.size)].Lock()
}

func (s *StripedLockSet) Unlock(key string) {
	s.locks[keyToID(key, s.size)].Unlock()
}

func (s *StripedLockSet) GlobalLock() {
	for i := range s.locks {
		s.locks[i].Lock()
	}
}

func (s *StripedLockSet) GlobalUnlock() {
	for i := range s.locks {
		s.locks[i].Unlock()
	}
}

func keyToID(key string, sliceSize int32) uint32 {
	h := fnv.New32()
	h.Write([]byte(key))
	return h.Sum32() % uint32(sliceSize)
}
