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

package lockingpromise

import (
	"sync"

	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise"
)

// lockingPromise implements LockingMutable based on a condition
// variable.  This implementation tracks active goroutines: the given
// counter is decremented for a goroutine waiting for this varible to
// be set and incremented when such a goroutine is unblocked.
type lockingPromise struct {
	lock          sync.Locker
	cond          sync.Cond
	activeCounter counter.GoRoutineCounter // counter of active goroutines
	waitingCount  int                      // number of goroutines idle due to this mutable being unset
	isSet         bool
	value         interface{}
}

var _ promise.LockingMutable = &lockingPromise{}

// NewLockingPromise makes a new promise.LockingMutable
func NewLockingPromise(lock sync.Locker, activeCounter counter.GoRoutineCounter) promise.LockingMutable {
	return &lockingPromise{
		lock:          lock,
		cond:          *sync.NewCond(lock),
		activeCounter: activeCounter,
	}
}

func (lp *lockingPromise) Set(value interface{}) {
	lp.lock.Lock()
	defer lp.lock.Unlock()
	lp.SetLocked(value)
}

func (lp *lockingPromise) Get() interface{} {
	lp.lock.Lock()
	defer lp.lock.Unlock()
	return lp.GetLocked()
}

func (lp *lockingPromise) SetLocked(value interface{}) {
	lp.isSet = true
	lp.value = value
	if lp.waitingCount > 0 {
		lp.activeCounter.Add(lp.waitingCount)
		lp.waitingCount = 0
		lp.cond.Broadcast()
	}
}

func (lp *lockingPromise) GetLocked() interface{} {
	if !lp.isSet {
		lp.waitingCount++
		lp.activeCounter.Add(-1)
		lp.cond.Wait()
	}
	return lp.value
}
