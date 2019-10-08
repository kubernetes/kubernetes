/*
Copyright 2016 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"

	"github.com/golang/groupcache/lru"
)

func expectEntry(t *testing.T, c *LRUExpireCache, key lru.Key, value interface{}) {
	result, ok := c.Get(key)
	if !ok || result != value {
		t.Errorf("Expected cache[%v]: %v, got %v", key, value, result)
	}
}

func expectNotEntry(t *testing.T, c *LRUExpireCache, key lru.Key) {
	if result, ok := c.Get(key); ok {
		t.Errorf("Expected cache[%v] to be empty, got %v", key, result)
	}
}

func TestSimpleGet(t *testing.T) {
	c := NewLRUExpireCache(10)
	c.Add("long-lived", "12345", 10*time.Hour)
	expectEntry(t, c, "long-lived", "12345")
}

func TestExpiredGet(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Time{})
	c := NewLRUExpireCacheWithClock(10, fakeClock)
	c.Add("short-lived", "12345", 1*time.Millisecond)
	// ensure the entry expired
	fakeClock.Step(2 * time.Millisecond)
	expectNotEntry(t, c, "short-lived")
}

func TestLRUOverflow(t *testing.T) {
	c := NewLRUExpireCache(4)
	c.Add("elem1", "1", 10*time.Hour)
	c.Add("elem2", "2", 10*time.Hour)
	c.Add("elem3", "3", 10*time.Hour)
	c.Add("elem4", "4", 10*time.Hour)
	c.Add("elem5", "5", 10*time.Hour)
	expectNotEntry(t, c, "elem1")
	expectEntry(t, c, "elem2", "2")
	expectEntry(t, c, "elem3", "3")
	expectEntry(t, c, "elem4", "4")
	expectEntry(t, c, "elem5", "5")
}

type joiningTestFramework struct {
	clock *clock.FakeClock
	cache *LRUExpireCache
}

func newJoiningTest(maxEntries int) *joiningTestFramework {
	j := &joiningTestFramework{
		clock: clock.NewFakeClock(time.Time{}),
	}
	j.cache = NewLRUExpireCacheWithClock(maxEntries, j.clock)
	return j
}

type blockedComputeFunc struct {
	framework *joiningTestFramework

	key   string
	value string

	// if a race is expected, the winning compute function increments this.
	winCount int64

	computeTTL    time.Duration
	cacheEntryTTL time.Duration

	callCount  int64
	blockUntil chan struct{}
}

func newBCF(j *joiningTestFramework, k, v string, computeTTL, entryTTL time.Duration) *blockedComputeFunc {
	return &blockedComputeFunc{
		framework:     j,
		key:           k,
		value:         v,
		computeTTL:    computeTTL,
		cacheEntryTTL: entryTTL,
		blockUntil:    make(chan struct{}),
	}
}

func (b *blockedComputeFunc) done() {
	close(b.blockUntil)
}

func (b *blockedComputeFunc) compute(notifyOnBegin chan<- struct{}) func(abort <-chan time.Time) (interface{}, time.Duration) {
	return func(abort <-chan time.Time) (interface{}, time.Duration) {
		notifyOnBegin <- struct{}{}
		// fmt.Printf("compute[%v=%v] begin: %v\n", b.key, b.value, b.framework.clock.Now())
		atomic.AddInt64(&b.callCount, 1)

		// Make sure that if both channels are waiting when we get
		// here, the abort channel is selected.
		select {
		case <-abort:
			// fmt.Printf("compute[%v=%v] aborting: %v\n", b.key, b.value, b.framework.clock.Now())
			return nil, 0
		default:
		}

		select {
		case <-b.blockUntil:
			// fmt.Printf("compute[%v=%v] unblocked: %v\n", b.key, b.value, b.framework.clock.Now())
			return b.value, b.cacheEntryTTL
		case <-abort:
			// fmt.Printf("compute[%v=%v] aborting: %v\n", b.key, b.value, b.framework.clock.Now())
			return nil, 0
		}
	}
}

// returns true if a thread was actually unblocked
func (b *blockedComputeFunc) tryUnblockOne() bool {
	select {
	case b.blockUntil <- struct{}{}:
		return true
	default:
	}
	// This can happen if all compute() funcs have exited, OR if they
	// haven't gotten far enough to wait on the channel yet -- we have to
	// count starts before trying unblocks to avoid this..
	return false
}

// unblockAtLeast attempts to unblock calls, and stops when it unblocks n
// calls, or after 100 attempts.
// If n is zero, make 100 attempts (for verifying no unexpected threads were pending).
//
// Returns the number of unblocked calls.
func (b *blockedComputeFunc) unblockAtLeast(n int) int {
	got := 0
	for i := 0; i < n+2; i++ {
		if b.tryUnblockOne() {
			got++
			if got == n {
				break
			}
		}
	}
	return got
}

// checkOne calls GetOrWait.
// wg is incremented and wg.Done() is called after the GetOrWait call has exited.
// One signal is sent down notifyOnBegin, either when the computeFunc enters,
// or immediately when GetOrWait finishes (if the computeFunc wasn't called due
// to a cache hit). eval() is called on the result of GetOrWait.
func (b *blockedComputeFunc) checkOne(wg *sync.WaitGroup, notifyOnBegin chan<- struct{}, eval func(interface{}, bool)) {
	wg.Add(1)
	ensureStarted := make(chan struct{}, 2) // might have 2 entries written to it, and we don't wish to block.
	go func() {
		defer wg.Done()
		v, ok := b.framework.cache.GetOrWait(b.key, b.compute(ensureStarted), b.computeTTL)

		// we don't know if we'll enter the compute func or not, so ping this just in case.
		ensureStarted <- struct{}{}
		// fmt.Println("pinging ensureStarted redundantly")
		eval(v, ok)
	}()
	go func() {
		<-ensureStarted
		// fmt.Println("definitely GetOrWait started somehow")
		notifyOnBegin <- struct{}{}
	}()
}

func (b *blockedComputeFunc) expectExactly(t *testing.T) func(interface{}, bool) {
	return func(v interface{}, ok bool) {
		if !ok {
			t.Errorf("unexpected cache miss")
		}
		if v != b.value {
			t.Errorf("unexpected contents: %v", v)
		}
	}
}

func (b *blockedComputeFunc) expectMiss(t *testing.T) func(interface{}, bool) {
	return func(v interface{}, ok bool) {
		if ok {
			t.Errorf("expected cache miss but got %v", v)
		}
	}
}

func (b *blockedComputeFunc) expectRace(t *testing.T) func(interface{}, bool) {
	return func(v interface{}, ok bool) {
		if ok && v == b.value {
			atomic.AddInt64(&b.winCount, 1)
		}
	}
}

func (b *blockedComputeFunc) Calls() int {
	return int(atomic.LoadInt64(&b.callCount))
}

func TestJoinedGet(t *testing.T) {
	t.Parallel()
	j := newJoiningTest(10)

	cf := newBCF(j, "joined", "12345", 2*time.Second, 10*time.Second)
	defer cf.done()

	var wg sync.WaitGroup
	startCounter := make(chan struct{}, 2)

	cf.checkOne(&wg, startCounter, cf.expectExactly(t))
	j.clock.Step(500 * time.Millisecond)

	cf.checkOne(&wg, startCounter, cf.expectExactly(t))
	j.clock.Step(1500 * time.Millisecond)

	<-startCounter

	// The two calls to unblockAtLeast are to notice if a second compute function was started.
	unblocked := cf.unblockAtLeast(1)
	<-startCounter
	unblocked += cf.unblockAtLeast(0)

	if unblocked != 1 {
		t.Errorf("expected to unblock %v waiting compute() calls, but unblocked %v", 1, unblocked)
	}

	wg.Wait()

	if calls := cf.Calls(); calls != 1 {
		t.Errorf("saw %v calls", calls)
	}

	expectEntry(t, j.cache, "joined", "12345")
}

func TestJoinedGetTimeout(t *testing.T) {
	var table = []struct {
		wait            time.Duration
		expectTimeout   bool
		expectUnblocked int
		expectCalls     int
	}{
		{1999 * time.Millisecond, false, 1, 1},
		{2001 * time.Millisecond, true, 0, 1},
	}
	for _, tt := range table {
		t.Run("", func(t *testing.T) {
			j := newJoiningTest(10)

			cf := newBCF(j, "joined", "12345", 2*time.Second, 10*time.Second)
			defer cf.done()

			var wg sync.WaitGroup
			startCounter := make(chan struct{}, 2)

			if tt.expectTimeout {
				cf.checkOne(&wg, startCounter, cf.expectMiss(t))
			} else {
				cf.checkOne(&wg, startCounter, cf.expectExactly(t))
			}
			<-startCounter
			j.clock.Step(tt.wait)

			unblocked := cf.unblockAtLeast(1)
			wg.Wait()

			if tt.expectTimeout {
				expectNotEntry(t, j.cache, "joined")
			} else {
				expectEntry(t, j.cache, "joined", "12345")
			}

			if e, a := tt.expectUnblocked, unblocked; e != a {
				t.Errorf("expected to unblock %v waiting compute() calls, but unblocked %v", e, a)
			}

			if e, a := tt.expectCalls, cf.Calls(); e != a {
				t.Errorf("expected %v calls but saw %v calls", e, a)
			}
		})
	}
}

func TestRegularGetDuringJoinedGet(t *testing.T) {
	j := newJoiningTest(10)

	cf := newBCF(j, "joined", "12345", 2*time.Second, 10*time.Second)
	defer cf.done()

	var wg sync.WaitGroup
	startCounter := make(chan struct{}, 2)

	cf.checkOne(&wg, startCounter, cf.expectExactly(t))
	<-startCounter
	j.clock.Step(1800 * time.Millisecond)

	// An entry is in the cache, but a regular get doesn't block on it.
	expectNotEntry(t, j.cache, "joined")

	unblocked := cf.unblockAtLeast(1)
	wg.Wait()

	expectEntry(t, j.cache, "joined", "12345")

	if e, a := 1, unblocked; e != a {
		t.Errorf("expected to unblock %v waiting compute() calls, but unblocked %v", e, a)
	}

	if e, a := 1, cf.Calls(); e != a {
		t.Errorf("expected %v calls but saw %v calls", e, a)
	}
}

func TestEvictDuringJoinedGet(t *testing.T) {
	j := newJoiningTest(3)

	cf := newBCF(j, "joined", "12345", 2*time.Second, 10*time.Second)
	defer cf.done()

	var wg sync.WaitGroup
	startCounter := make(chan struct{}, 2)

	cf.checkOne(&wg, startCounter, cf.expectExactly(t))
	<-startCounter
	j.clock.Step(1800 * time.Millisecond)

	j.cache.Add("elem1", "1", 10*time.Hour)
	j.cache.Add("elem2", "2", 10*time.Hour)
	j.cache.Add("elem3", "3", 10*time.Hour)

	// Because the first entry has been evicted already, this will cause a
	// second call and not be joined.
	cf.checkOne(&wg, startCounter, cf.expectExactly(t))
	<-startCounter

	unblocked := cf.unblockAtLeast(2)
	wg.Wait()

	expectEntry(t, j.cache, "joined", "12345")

	if e, a := 2, unblocked; e != a {
		t.Errorf("expected to unblock %v waiting compute() calls, but unblocked %v", e, a)
	}

	if e, a := 2, cf.Calls(); e != a {
		t.Errorf("expected %v calls but saw %v calls", e, a)
	}
}

func TestJoinedThrashing(t *testing.T) {
	j := newJoiningTest(20)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	startCounter := make(chan struct{}, 1000)
	wg.Add(2)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			case <-startCounter:
			}
		}
	}()
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				j.clock.Step(1700 * time.Millisecond)
			}
		}
	}()

	thrash := func(cf *blockedComputeFunc) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer cf.done()
			unblocked := 0
			for {
				unblocked += cf.unblockAtLeast(1)
				select {
				case <-stop:
					return
				default:
				}
			}
			t.Logf("unblocked %v total threads", unblocked)
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				var wg2 sync.WaitGroup
				for i := 0; i < 5; i++ {
					cf.checkOne(&wg2, startCounter, cf.expectRace(t))
				}
				wg2.Wait()
				select {
				case <-stop:
					return
				default:
				}
			}
		}()
	}

	for i := 0; i < 100; i++ {
		str := fmt.Sprintf("%v", i)
		thrash(newBCF(j, "race_"+str, "12345", 2*time.Second, 10*time.Second))
		thrash(newBCF(j, "norace_"+str, "12345", 2*time.Second, 2*time.Second))
		thrash(newBCF(j, "race_"+str, "54321", 2*time.Second, 2*time.Second))
	}

	time.Sleep(5 * time.Second)
	close(stop)
	wg.Wait()
}
