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

// promisoid is the data and behavior common to all the promise-like
// abstractions implemented here.  This implementation is based on a
// condition variable.  This implementation tracks active goroutines:
// the given counter is decremented for a goroutine waiting for this
// varible to be set and incremented when such a goroutine is
// unblocked.
type promisoid struct {
	lock          sync.Locker
	cond          sync.Cond
	activeCounter counter.GoRoutineCounter // counter of active goroutines
	waitingCount  int                      // number of goroutines idle due to this being unset
	isSet         bool
	value         interface{}
}

func (pr *promisoid) Get() interface{} {
	pr.lock.Lock()
	defer pr.lock.Unlock()
	return pr.GetLocked()
}

func (pr *promisoid) GetLocked() interface{} {
	if !pr.isSet {
		pr.waitingCount++
		pr.activeCounter.Add(-1)
		pr.cond.Wait()
	}
	return pr.value
}

func (pr *promisoid) IsSet() bool {
	pr.lock.Lock()
	defer pr.lock.Unlock()
	return pr.IsSetLocked()
}

func (pr *promisoid) IsSetLocked() bool {
	return pr.isSet
}

func (pr *promisoid) SetLocked(value interface{}) {
	pr.isSet = true
	pr.value = value
	if pr.waitingCount > 0 {
		pr.activeCounter.Add(pr.waitingCount)
		pr.waitingCount = 0
		pr.cond.Broadcast()
	}
}

type writeOnce struct {
	promisoid
}

var _ promise.LockingWriteOnce = &writeOnce{}

// NewWriteOnce makes a new promise.LockingWriteOnce
func NewWriteOnce(lock sync.Locker, activeCounter counter.GoRoutineCounter) promise.LockingWriteOnce {
	return &writeOnce{promisoid{
		lock:          lock,
		cond:          *sync.NewCond(lock),
		activeCounter: activeCounter,
	}}
}

func (wr *writeOnce) Set(value interface{}) bool {
	wr.lock.Lock()
	defer wr.lock.Unlock()
	return wr.SetLocked(value)
}

func (wr *writeOnce) SetLocked(value interface{}) bool {
	if wr.isSet {
		return false
	}
	wr.promisoid.SetLocked(value)
	return true
}

type writeMultiple struct {
	promisoid
}

var _ promise.LockingWriteMultiple = &writeMultiple{}

// NewWriteMultiple makes a new promise.LockingWriteMultiple
func NewWriteMultiple(lock sync.Locker, activeCounter counter.GoRoutineCounter) promise.LockingWriteMultiple {
	return &writeMultiple{promisoid{
		lock:          lock,
		cond:          *sync.NewCond(lock),
		activeCounter: activeCounter,
	}}
}

func (wr *writeMultiple) Set(value interface{}) {
	wr.lock.Lock()
	defer wr.lock.Unlock()
	wr.SetLocked(value)
}
