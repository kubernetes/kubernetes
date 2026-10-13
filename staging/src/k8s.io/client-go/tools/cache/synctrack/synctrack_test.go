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

	"testing"
)

func testSingleFileFuncs() (upstreamHasSynced func(), start func(), finished func(), hasSynced func() bool, synced <-chan struct{}) {
	tracker := NewSingleFileTracker("")
	return tracker.UpstreamHasSynced, tracker.Start, tracker.Finished, tracker.HasSynced, tracker.Done()
}

func testAsyncFuncs() (upstreamHasSynced func(), start func(), finished func(), hasSynced func() bool, synced <-chan struct{}) {
	tracker := NewAsyncTracker[string]("")
	return tracker.UpstreamHasSynced, func() { tracker.Start("key") }, func() { tracker.Finished("key") }, tracker.HasSynced, tracker.Done()
}

func TestBasicLogic(t *testing.T) {
	table := []struct {
		name      string
		construct func() (func(), func(), func(), func() bool, <-chan struct{})
	}{
		{"SingleFile", testSingleFileFuncs},
		{"Async", testAsyncFuncs},
	}

	for _, entry := range table {
		t.Run(entry.name, func(t *testing.T) {
			table := []struct {
				synced             bool
				syncedBeforeFinish bool

				start        bool
				finish       bool
				expectSynced bool
			}{
				{false, false, true, true, false},
				{true, false, true, false, false},
				{true, true, true, false, false},
				{false, false, true, false, false},
				{true, false, true, true, true},
				{true, true, true, true, true},
			}
			for _, tt := range table {
				upstreamHasSynced, start, finished, hasSynced, synced := entry.construct()
				syncedDone := func() bool {
					select {
					case <-synced:
						return true
					default:
						return false
					}
				}

				if hasSynced() {
					t.Errorf("for %#v got HasSynced() true before start (wanted false)", tt)
				}
				if syncedDone() {
					t.Errorf("for %#v got Done() true before start (wanted false)", tt)
				}

				if tt.start {
					start()
				}

				if hasSynced() {
					t.Errorf("for %#v got HasSynced() true after start (wanted false)", tt)
				}
				if syncedDone() {
					t.Errorf("for %#v got Done() true after start (wanted false)", tt)
				}

				// "upstream has synced" may occur before or after finished, but not before start.
				if tt.synced && tt.syncedBeforeFinish {
					upstreamHasSynced()
					if hasSynced() {
						t.Errorf("for %#v got HasSynced() true after upstreamHasSynced and before finish (wanted false)", tt)
					}
					if syncedDone() {
						t.Errorf("for %#v got Done() true after upstreamHasSynced and before finish (wanted false)", tt)
					}
				}
				if tt.finish {
					finished()
				}
				if tt.synced && !tt.syncedBeforeFinish {
					if hasSynced() {
						t.Errorf("for %#v got HasSynced() true after finish and before upstreamHasSynced (wanted false)", tt)
					}
					if syncedDone() {
						t.Errorf("for %#v got Done() true after finish and before upstreamHasSynced (wanted false)", tt)
					}
					upstreamHasSynced()
				}
				if e, a := tt.expectSynced, hasSynced(); e != a {
					t.Errorf("for %#v got HasSynced() %v (wanted %v)", tt, a, e)
				}
				if e, a := tt.expectSynced, syncedDone(); e != a {
					t.Errorf("for %#v got Done() %v (wanted %v)", tt, a, e)
				}

				select {
				case <-synced:
					if !tt.expectSynced {
						t.Errorf("for %#v got done (wanted not done)", tt)
					}
				default:
					if tt.expectSynced {
						t.Errorf("for %#v got done (wanted not done)", tt)
					}
				}
			}
		})
	}
}

func TestAsyncLocking(t *testing.T) {
	aft := NewAsyncTracker[int]("")

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
	aft.UpstreamHasSynced()
	if !aft.HasSynced() {
		t.Errorf("async tracker must have made a threading error?")
	}

}

func TestSingleFileCounting(t *testing.T) {
	sft := NewSingleFileTracker("")

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
	sft.UpstreamHasSynced()

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
		synced            bool
		syncedBeforeStops bool

		starts       int
		stops        int
		expectSynced bool
	}{
		{false, false, 1, 1, false},
		{true, false, 1, 0, false},
		{true, true, 1, 0, false},
		{false, false, 1, 0, false},
		{true, false, 1, 1, true},
		{true, true, 1, 1, true},
	}
	for _, tt := range table {
		sft := NewSingleFileTracker("")
		for i := 0; i < tt.starts; i++ {
			sft.Start()
		}
		// "upstream has synced" may occur before or after finished, but not before start.
		if tt.synced && tt.syncedBeforeStops {
			sft.UpstreamHasSynced()
		}
		for i := 0; i < tt.stops; i++ {
			sft.Finished()
		}
		if tt.synced && !tt.syncedBeforeStops {
			sft.UpstreamHasSynced()
		}
		got := sft.HasSynced()
		if e, a := tt.expectSynced, got; e != a {
			t.Errorf("for %#v got %v (wanted %v)", tt, a, e)
		}
	}

}
