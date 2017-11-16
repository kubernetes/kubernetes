/*
Copyright 2017 The Kubernetes Authors.

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

package waitgroup

import (
	"fmt"
	"sync"
)

// SafeWaitGroup must not be copied after first use.
type SafeWaitGroup struct {
	wg sync.WaitGroup
	mu sync.RWMutex
	// wait indicate whether Wait is called, if true,
	// then any Add with positive delta will return error.
	wait bool
}

// Add adds delta, which may be negative, similar to sync.WaitGroup.
// If Add with a positive delta happens after Wait, it will return error,
// which prevent unsafe Add.
func (wg *SafeWaitGroup) Add(delta int) error {
	wg.mu.RLock()
	defer wg.mu.RUnlock()
	if wg.wait && delta > 0 {
		return fmt.Errorf("add with postive delta after Wait is forbidden")
	}
	wg.wg.Add(delta)
	return nil
}

// Done decrements the WaitGroup counter.
func (wg *SafeWaitGroup) Done() {
	wg.wg.Done()
}

// Wait blocks until the WaitGroup counter is zero.
func (wg *SafeWaitGroup) Wait() {
	wg.mu.Lock()
	wg.wait = true
	wg.mu.Unlock()
	wg.wg.Wait()
}
