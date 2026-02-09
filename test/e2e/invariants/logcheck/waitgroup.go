/*
Copyright The Kubernetes Authors.

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

package logcheck

import (
	"sync"
)

// waitGroup, in contrast to sync.WaitGroup, supports races between
// go and wait. Once wait is called, no new goroutines are started.
type waitGroup struct {
	mutex sync.Mutex
	cond  *sync.Cond

	shuttingDown bool
	running      int
}

func (wg *waitGroup) wait() {
	wg.mutex.Lock()
	defer wg.mutex.Unlock()

	wg.shuttingDown = true
	for wg.running > 0 {
		wg.cond.Wait()
	}
}

func (wg *waitGroup) add(n int) bool {
	wg.mutex.Lock()
	defer wg.mutex.Unlock()

	if wg.shuttingDown {
		return false
	}

	wg.running += n
	return true
}

func (wg *waitGroup) done() {
	wg.mutex.Lock()
	defer wg.mutex.Unlock()

	wg.running--
	wg.cond.Broadcast()
}

// goIfNotShuttingDown executes the callback in a goroutine if the wait group
// is not already shutting down. It always calls the cleanup callback once, either way.
func (wg *waitGroup) goIfNotShuttingDown(cleanup, cb func()) {
	wg.mutex.Lock()
	defer wg.mutex.Unlock()

	if cleanup == nil {
		cleanup = func() {}
	}

	if wg.shuttingDown {
		// Clean up directly.
		cleanup()
		return
	}

	wg.running++
	go func() {
		defer wg.done()
		defer cleanup()
		cb()
	}()
}

func newWaitGroup() *waitGroup {
	var wg waitGroup
	wg.cond = sync.NewCond(&wg.mutex)
	return &wg
}
