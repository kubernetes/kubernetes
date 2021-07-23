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
	cond          sync.Cond
	activeCounter counter.GoRoutineCounter // counter of active goroutines
	waitingCount  int                      // number of goroutines idle due to this being unset
	isSet         bool
	value         interface{}
}

var _ WriteOnce = &promise{}

// NewWriteOnce makes a new promise.LockingWriteOnce
func NewWriteOnce(lock sync.Locker, activeCounter counter.GoRoutineCounter) WriteOnce {
	return &promise{
		cond:          *sync.NewCond(lock),
		activeCounter: activeCounter,
	}
}

func (p *promise) Get() interface{} {
	if !p.isSet {
		p.waitingCount++
		p.activeCounter.Add(-1)
		p.cond.Wait()
	}
	return p.value
}

func (p *promise) IsSet() bool {
	return p.isSet
}

func (p *promise) Set(value interface{}) bool {
	if p.isSet {
		return false
	}
	p.isSet = true
	p.value = value
	if p.waitingCount > 0 {
		p.activeCounter.Add(p.waitingCount)
		p.waitingCount = 0
		p.cond.Broadcast()
	}
	return true
}
