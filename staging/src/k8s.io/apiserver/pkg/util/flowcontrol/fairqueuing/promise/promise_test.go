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
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/clock"
)

func TestLockingWriteOnce(t *testing.T) {
	now := time.Now()
	clock, counter := clock.NewFakeEventClock(now, 0, nil)
	var lock sync.Mutex
	wr := NewWriteOnce(&lock, counter)
	var gots int32
	var got atomic.Value
	counter.Add(1)
	go func() {
		lock.Lock()
		defer lock.Unlock()

		got.Store(wr.Get())
		atomic.AddInt32(&gots, 1)
		counter.Add(-1)
	}()
	clock.Run(nil)
	time.Sleep(time.Second)
	if atomic.LoadInt32(&gots) != 0 {
		t.Error("Get returned before Set")
	}
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.IsSet() {
			t.Error("IsSet before Set")
		}
	}()
	aval := &now
	func() {
		lock.Lock()
		defer lock.Unlock()
		if !wr.Set(aval) {
			t.Error("Set() returned false")
		}
	}()
	clock.Run(nil)
	time.Sleep(time.Second)
	if atomic.LoadInt32(&gots) != 1 {
		t.Error("Get did not return after Set")
	}
	if got.Load() != aval {
		t.Error("Get did not return what was Set")
	}
	func() {
		lock.Lock()
		defer lock.Unlock()
		if !wr.IsSet() {
			t.Error("IsSet()==false after Set")
		}
	}()
	counter.Add(1)
	go func() {
		lock.Lock()
		defer lock.Unlock()

		got.Store(wr.Get())
		atomic.AddInt32(&gots, 1)
		counter.Add(-1)
	}()
	clock.Run(nil)
	time.Sleep(time.Second)
	if atomic.LoadInt32(&gots) != 2 {
		t.Error("Second Get did not return immediately")
	}
	if got.Load() != aval {
		t.Error("Second Get did not return what was Set")
	}
	func() {
		lock.Lock()
		defer lock.Unlock()
		if !wr.IsSet() {
			t.Error("IsSet()==false after second Get")
		}
	}()
	later := time.Now()
	bval := &later
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.Set(bval) {
			t.Error("second Set() returned true")
		}
	}()
	func() {
		lock.Lock()
		defer lock.Unlock()
		if !wr.IsSet() {
			t.Error("IsSet() returned false after second set")
		} else if wr.Get() != aval {
			t.Error("Get() after second Set returned wrong value")
		}
	}()
}
