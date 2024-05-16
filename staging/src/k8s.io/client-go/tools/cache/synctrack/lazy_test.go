/*
Copyright 2023 The Kubernetes Authors.

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

package synctrack_test

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/client-go/tools/cache/synctrack"
)

func TestLazy(t *testing.T) {
	var reality int64
	var z synctrack.Lazy[int64]

	z.Evaluate = func() (int64, error) {
		return atomic.LoadInt64(&reality), nil
	}

	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(delay time.Duration) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				t.Helper()
				set := atomic.AddInt64(&reality, 1)
				z.Notify()
				got, err := z.Get()
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if got < set {
					t.Errorf("time went backwards. %v vs %v", got, set)
				}
				time.Sleep(delay)
			}
		}((1 + time.Duration(i%3)) * time.Microsecond)
	}

	wg.Wait()
}

func TestLazyThroughput(t *testing.T) {
	var reality int64
	var z synctrack.Lazy[int64]
	var totalWait int64
	z.Evaluate = func() (int64, error) {
		got := atomic.LoadInt64(&reality)
		time.Sleep(11 * time.Millisecond)
		return got, nil
	}

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		notifies := 0
		tt := time.NewTicker(10 * time.Millisecond)
		for {
			<-tt.C
			atomic.AddInt64(&reality, 1)
			z.Notify()
			notifies++
			if notifies >= 100 {
				tt.Stop()
				return
			}
			wg.Add(1)
			go func() {
				t.Helper()
				defer wg.Done()
				start := time.Now()
				z.Get()
				d := time.Since(start)
				atomic.AddInt64(&totalWait, int64(d))
			}()
		}
	}()

	wg.Wait()

	twd := time.Duration(totalWait)

	if twd > 3*time.Second {
		t.Errorf("total wait was: %v; par would be ~1s", twd)
	}

}

// sequence is for controlling the order various lines of code execute in.
// Replaces a bunch of time.Sleep() calls that would certainly be flaky.
type sequence []sync.WaitGroup

func newSequence(n int) sequence {
	s := make(sequence, n)
	for i := range s {
		s[i].Add(1)
	}
	return s
}

func (s sequence) Start() { s[0].Done() }

func (s sequence) Step(n int) {
	s[n].Wait()
	if n+1 < len(s) {
		s[n+1].Done()
	}
}

// asyncGet runs a goroutine to do the get so it doesn't block.
func asyncGet[T any](t *testing.T, seq sequence, z *synctrack.Lazy[T], pre, post int) func() T {
	var wg sync.WaitGroup
	var val T
	wg.Add(1)
	go func() {
		defer wg.Done()
		t.Helper()
		var err error
		seq.Step(pre)
		val, err = z.Get()
		seq.Step(post)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}()
	return func() T { wg.Wait(); return val }
}

func TestLazySlowEval(t *testing.T) {
	// This tests the case where the first invocation of eval finishes
	// after a subseqent invocation. The old value should not be put into
	// the cache and returned. Nor should eval be called an extra time to
	// correct the old value having been placed into the cache.

	seq := newSequence(10)

	var getCount int64
	var z synctrack.Lazy[int64]

	z.Evaluate = func() (int64, error) {
		count := atomic.AddInt64(&getCount, 1)
		if count == 1 {
			seq.Step(1)
			seq.Step(6)
		} else if count > 2 {
			t.Helper()
			t.Errorf("Eval called extra times. count=%v", count)
		} else {
			seq.Step(4)
		}
		return time.Now().UnixNano(), nil
	}

	seq.Start()

	getA := asyncGet(t, seq, &z, 0, 7)

	seq.Step(2)
	z.Notify()

	getB := asyncGet(t, seq, &z, 3, 5)

	getC := asyncGet(t, seq, &z, 8, 9)

	a, b, c := getA(), getB(), getC()
	if a < b {
		t.Errorf("failed to create the test condition")
	}
	if b != c && c == a {
		t.Errorf("wrong value was cached")
	}
}

func TestLazySlowEval2(t *testing.T) {
	// This tests the case where the first invocation of eval finishes
	// before a subseqent invocation. The old value should be overwritten.
	// Eval should not be called an extra time to correct the wrong value
	// having been placed into the cache.

	seq := newSequence(11)

	var getCount int64
	var z synctrack.Lazy[int64]

	z.Evaluate = func() (int64, error) {
		count := atomic.AddInt64(&getCount, 1)
		if count == 1 {
			seq.Step(1)
			seq.Step(5)
		} else if count > 2 {
			t.Helper()
			t.Errorf("Eval called extra times. count=%v", count)
		} else {
			seq.Step(4)
			seq.Step(7)
		}
		return time.Now().UnixNano(), nil
	}

	seq.Start()

	getA := asyncGet(t, seq, &z, 0, 6)

	seq.Step(2)

	z.Notify()

	getB := asyncGet(t, seq, &z, 3, 8)

	getC := asyncGet(t, seq, &z, 9, 10)

	a, b, c := getA(), getB(), getC()
	if a > b {
		t.Errorf("failed to create the test condition")
	}
	if b != c && c == a {
		t.Errorf("wrong value was cached")
	}
}

func TestLazyOnlyOnce(t *testing.T) {
	// This demonstrates that multiple Gets don't cause multiple Evaluates.

	seq := newSequence(8)

	var getCount int64
	var z synctrack.Lazy[int64]

	z.Evaluate = func() (int64, error) {
		count := atomic.AddInt64(&getCount, 1)
		if count == 1 {
			seq.Step(1)
			seq.Step(4)
		} else if count > 1 {
			t.Helper()
			t.Errorf("Eval called extra times. count=%v", count)
		}
		return time.Now().UnixNano(), nil
	}

	seq.Start()

	z.Notify()

	getA := asyncGet(t, seq, &z, 0, 5)
	getB := asyncGet(t, seq, &z, 2, 6)
	getC := asyncGet(t, seq, &z, 3, 7)

	a, b, c := getA(), getB(), getC()
	if a > b {
		t.Errorf("failed to create the test condition")
	}
	if b != c && c == a {
		t.Errorf("wrong value was cached")
	}
}

func TestLazyError(t *testing.T) {
	var succeed bool
	var z synctrack.Lazy[bool]
	z.Evaluate = func() (bool, error) {
		if succeed {
			return true, nil
		} else {
			return false, errors.New("deliberate fail")
		}
	}

	if _, err := z.Get(); err == nil {
		t.Fatalf("expected error")
	}
	// Note: no notify, proving the error was not cached
	succeed = true
	if _, err := z.Get(); err != nil {
		t.Fatalf("unexpected error")
	}
}
