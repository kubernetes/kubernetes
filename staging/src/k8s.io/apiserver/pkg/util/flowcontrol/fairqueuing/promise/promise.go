/*
Copyright 2019 The Kubernetes Authors.

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

package promise

import (
	"fmt"
	"sync"

	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
)

// promise implements the promise.WriteOnce interface.
// This implementation is based on a condition variable.
// This implementation tracks active goroutines:
// the given counter is decremented for a goroutine waiting for this
// varible to be set and incremented when such a goroutine is
// unblocked.
type promise struct {
	lock sync.Mutex

	doneCh        chan struct{}
	activeCounter counter.GoRoutineCounter // counter of active goroutines
	waitingCount  int                      // number of goroutines idle due to this being unset

	isSet         bool
	value         interface{}
}

var _ WriteOnce = &promise{}

// NewWriteOnce makes a new promise.LockingWriteOnce
func NewWriteOnce(activeCounter counter.GoRoutineCounter) WriteOnce {
	return &promise{
		doneCh:       make(chan struct{}),
		activeCounter: activeCounter,
	}
}

// FIXME:
var timeoutErr = fmt.Errorf("timeout")

func (p *promise) WaitAndGet(stopCh <-chan struct{}) (interface{}, error) {
	isSet := func() bool {
		p.lock.Lock()
		defer p.lock.Unlock()
		if p.isSet {
			return true
		}

		p.waitingCount++
		p.activeCounter.Add(-1)
		return false
	}()
	if isSet {
		return p.value, nil
	}

	select {
	case <-p.doneCh:
		return p.value, nil
	case <-stopCh:
		p.lock.Lock()
		defer p.lock.Unlock()
		// Check if in the meantime value wasn't set (and thus the goroutine wasn't activated).
		select {
		case <-p.doneCh:
			return p.value, nil
		default:
		}
		// Now we know we really timed out. So we reactivate the goroutine.
		p.waitingCount--
		p.activeCounter.Add(1)
		return nil, timeoutErr
	}
}

func (p *promise) IsSet() bool {
	p.lock.Lock()
	defer p.lock.Unlock()
	return p.isSet
}

func (p *promise) Set(value interface{}) bool {
	p.lock.Lock()
	defer p.lock.Unlock()
	if p.isSet {
		return false
	}

	p.isSet = true
	p.value = value
	if p.waitingCount > 0 {
		p.activeCounter.Add(p.waitingCount)
		p.waitingCount = 0
	}
	close(p.doneCh)
	return true
}
