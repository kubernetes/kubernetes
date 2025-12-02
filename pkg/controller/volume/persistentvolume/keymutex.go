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

package persistentvolume

import "sync"

// keyMutex provides per-key mutual exclusion so work on the same object
// does not run concurrently while allowing different keys to proceed in
// parallel.
type keyMutex struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

// Lock returns a function that must be called to unlock the mutex for the given key.
func (km *keyMutex) Lock(key string) func() {
	km.mu.Lock()
	if km.locks == nil {
		km.locks = make(map[string]*sync.Mutex)
	}
	m, ok := km.locks[key]
	if !ok {
		m = &sync.Mutex{}
		km.locks[key] = m
	}
	km.mu.Unlock()

	m.Lock()
	return m.Unlock
}
