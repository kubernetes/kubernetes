/*
Copyright 2022 The Kubernetes Authors.

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

package synctrack

import (
	"strings"
	"sync"
	"time"

	"testing"
)

func testSingleFileFuncs(upstreamHasSynced func() bool) (start func(), finished func(), hasSynced func() bool) {
	tracker := SingleFileTracker{
		UpstreamHasSynced: upstreamHasSynced,
	}
	return tracker.Start, tracker.Finished, tracker.HasSynced
}

func testAsyncFuncs(upstreamHasSynced func() bool) (start func(), finished func(), hasSynced func() bool) {
	tracker := AsyncTracker[string]{
		UpstreamHasSynced: upstreamHasSynced,
	}
	return func() { tracker.Start("key") }, func() { tracker.Finished("key") }, tracker.HasSynced
}

func TestBasicLogic(t *testing.T) {
	table := []struct {
		name      string
		construct func(func() bool) (func(), func(), func() bool)
	}{
		{"SingleFile", testSingleFileFuncs},
		{"Async", testAsyncFuncs},
	}

	for _, entry := range table {
		t.Run(entry.name, func(t *testing.T) {
			table := []struct {
				synced       bool
				start        bool
				finish       bool
				expectSynced bool
			}{
				{false, true, true, false},
				{true, true, false, false},
				{false, true, false, false},
				{true, true, true, true},
			}
			for _, tt := range table {
				Start, Finished, HasSynced := entry.construct(func() bool { return tt.synced })
				if tt.start {
					Start()
				}
				if tt.finish {
					Finished()
				}
				got := HasSynced()
				if e, a := tt.expectSynced, got; e != a {
					t.Errorf("for %#v got %v (wanted %v)", tt, a, e)
				}
			}
		})
	}
}

func TestAsyncLocking(t *testing.T) {
	aft := AsyncTracker[int]{UpstreamHasSynced: func() bool { return true }}

	var wg sync.WaitGroup
	for _, i := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10} {
		wg.Add(1)
		go func(i int) {
			aft.Start(i)
			go func() {
				aft.Finished(i)
				wg.Done()
			}()
		}(i)
	}
	wg.Wait()
	if !aft.HasSynced() {
		t.Errorf("async tracker must have made a threading error?")
	}

}

func TestSingleFileCounting(t *testing.T) {
	sft := SingleFileTracker{UpstreamHasSynced: func() bool { return true }}

	for i := 0; i < 100; i++ {
		sft.Start()
	}
	if sft.HasSynced() {
		t.Fatal("Unexpectedly synced?")
	}
	for i := 0; i < 99; i++ {
		sft.Finished()
	}
	if sft.HasSynced() {
		t.Fatal("Unexpectedly synced?")
	}

	sft.Finished()
	if !sft.HasSynced() {
		t.Fatal("Unexpectedly not synced?")
	}

	// Calling an extra time will panic.
	func() {
		defer func() {
			x := recover()
			if x == nil {
				t.Error("no panic?")
				return
			}
			msg, ok := x.(string)
			if !ok {
				t.Errorf("unexpected panic value: %v", x)
				return
			}
			if !strings.Contains(msg, "negative counter") {
				t.Errorf("unexpected panic message: %v", msg)
				return
			}
		}()
		sft.Finished()
	}()

	// Negative counter still means it is synced
	if !sft.HasSynced() {
		t.Fatal("Unexpectedly not synced?")
	}
}

func TestSingleFile(t *testing.T) {
	table := []struct {
		synced       bool
		starts       int
		stops        int
		expectSynced bool
	}{
		{false, 1, 1, false},
		{true, 1, 0, false},
		{false, 1, 0, false},
		{true, 1, 1, true},
	}
	for _, tt := range table {
		sft := SingleFileTracker{UpstreamHasSynced: func() bool { return tt.synced }}
		for i := 0; i < tt.starts; i++ {
			sft.Start()
		}
		for i := 0; i < tt.stops; i++ {
			sft.Finished()
		}
		got := sft.HasSynced()
		if e, a := tt.expectSynced, got; e != a {
			t.Errorf("for %#v got %v (wanted %v)", tt, a, e)
		}
	}

}

func TestNoStaleValue(t *testing.T) {
	table := []struct {
		name      string
		construct func(func() bool) (func(), func(), func() bool)
	}{
		{"SingleFile", testSingleFileFuncs},
		{"Async", testAsyncFuncs},
	}

	for _, entry := range table {
		t.Run(entry.name, func(t *testing.T) {
			var lock sync.Mutex
			upstreamHasSynced := func() bool {
				lock.Lock()
				defer lock.Unlock()
				return true
			}

			Start, Finished, HasSynced := entry.construct(upstreamHasSynced)

			// Ordinarily the corresponding lock would be held and you wouldn't be
			// able to call this function at this point.
			if !HasSynced() {
				t.Fatal("Unexpectedly not synced??")
			}

			Start()
			if HasSynced() {
				t.Fatal("Unexpectedly synced??")
			}
			Finished()
			if !HasSynced() {
				t.Fatal("Unexpectedly not synced??")
			}

			// Now we will prove that if the lock is held, you can't get a false
			// HasSynced return.
			lock.Lock()

			// This goroutine calls HasSynced
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				if HasSynced() {
					t.Error("Unexpectedly synced??")
				}
			}()

			// This goroutine increments + unlocks. The sleep is to bias the
			// runtime such that the other goroutine usually wins (it needs to work
			// in both orderings, this one is more likely to be buggy).
			go func() {
				time.Sleep(time.Millisecond)
				Start()
				lock.Unlock()
			}()

			wg.Wait()
		})
	}

}
