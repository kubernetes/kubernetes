// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bundler

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"
)

func TestBundlerCount1(t *testing.T) {
	// Unbundled case: one item per bundle.
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleImmediate)
	b.BundleCountThreshold = 1
	b.DelayThreshold = time.Second

	for i := 0; i < 3; i++ {
		if err := b.Add(i, 1); err != nil {
			t.Fatal(err)
		}
	}
	b.Flush()
	got := handler.bundles()
	want := [][]int{{0}, {1}, {2}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("bundles: got %v, want %v", got, want)
	}
	// All bundles should have been handled "immediately": much less
	// than the delay threshold of 1s.
	tgot := quantizeTimes(handler.times(), 100*time.Millisecond)
	twant := []int{0, 0, 0}
	if !reflect.DeepEqual(tgot, twant) {
		t.Errorf("times: got %v, want %v", tgot, twant)
	}
}

func TestBundlerCount3(t *testing.T) {
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleImmediate)
	b.BundleCountThreshold = 3
	b.DelayThreshold = 100 * time.Millisecond
	// Add 8 items.
	// The first two bundles of 3 should both be handled quickly.
	// The third bundle of 2 should not be handled for about DelayThreshold ms.
	for i := 0; i < 8; i++ {
		if err := b.Add(i, 1); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(5 * b.DelayThreshold)
	// We should not need to close the bundler.

	bgot := handler.bundles()
	bwant := [][]int{{0, 1, 2}, {3, 4, 5}, {6, 7}}
	if !reflect.DeepEqual(bgot, bwant) {
		t.Errorf("bundles: got %v, want %v", bgot, bwant)
	}

	tgot := quantizeTimes(handler.times(), b.DelayThreshold)
	if len(tgot) != 3 || tgot[0] != 0 || tgot[1] != 0 || tgot[2] == 0 {
		t.Errorf("times: got %v, want [0, 0, non-zero]", tgot)
	}
}

// Test that items are handled correctly at roughly the right time with a "slow"
// handler (takes 300 milliseconds) and that the last bundle is automatically
// flushed.
func TestBundlerCountSlowHandler(t *testing.T) {
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleSlow)
	b.BundleCountThreshold = 3
	b.DelayThreshold = 500 * time.Millisecond
	// Add 10 items.
	for i := 0; i < 10; i++ {
		if err := b.Add(i, 1); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(4 * 300 * time.Millisecond)
	// We should not need to close the bundler.

	bgot := handler.bundles()
	bwant := [][]int{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9}}
	if !reflect.DeepEqual(bgot, bwant) {
		t.Errorf("bundles: got %v, want %v", bgot, bwant)
	}

	tgot := quantizeTimes(handler.times(), 100*time.Millisecond)
	// Should handle new bundle every 300 milliseconds, and last incomplete
	// bundle should get automatically flushed.
	twant := []int{0, 3, 6, 9}
	if !reflect.DeepEqual(tgot, twant) {
		t.Errorf("times: got %v, want [0, 0, non-zero]", tgot)
	}
}

func TestBundlerByteThreshold(t *testing.T) {
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleImmediate)
	b.BundleCountThreshold = 10
	b.BundleByteThreshold = 3
	// Increase the limit beyond the number of bundles we expect (3)
	// so that bundles get handled immediately after they cross the
	// threshold. Otherwise, the test is non-deterministic. With the default
	// HandlerLimit of 1, the 2nd and 3rd bundles may or may not be
	// combined based on how long it takes to handle the 1st bundle.
	b.HandlerLimit = 10
	add := func(i interface{}, s int) {
		if err := b.Add(i, s); err != nil {
			t.Fatal(err)
		}
	}

	add(1, 1)
	add(2, 2)
	// Hit byte threshold AND under HandlerLimit:
	// bundle = 1, 2
	add(3, 1)
	add(4, 1)
	add(5, 2)
	// Passed byte threshold AND under byte limit AND under HandlerLimit:
	// bundle = 3, 4, 5
	add(6, 1)
	b.Flush()
	bgot := handler.bundles()
	// We don't care about the order they were handled in. We just want
	// to test that crossing the threshold triggered handling.
	sort.Slice(bgot, func(i, j int) bool {
		return bgot[i][0] < bgot[j][0]
	})
	bwant := [][]int{{1, 2}, {3, 4, 5}, {6}}
	if !reflect.DeepEqual(bgot, bwant) {
		t.Errorf("bundles: got %v, want %v", bgot, bwant)
	}
	tgot := quantizeTimes(handler.times(), b.DelayThreshold)
	twant := []int{0, 0, 0}
	if !reflect.DeepEqual(tgot, twant) {
		t.Errorf("times: got %v, want %v", tgot, twant)
	}
}

func TestBundlerLimit(t *testing.T) {
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleImmediate)
	b.BundleCountThreshold = 10
	b.BundleByteLimit = 3
	add := func(i interface{}, s int) {
		if err := b.Add(i, s); err != nil {
			t.Fatal(err)
		}
	}

	add(1, 1)
	add(2, 2)
	// Hit byte limit: bundle = 1, 2
	add(3, 1)
	add(4, 1)
	add(5, 2)
	// Exceeded byte limit: bundle = 3, 4
	add(6, 2)
	// Exceeded byte limit: bundle = 5
	b.Flush()
	bgot := handler.bundles()
	bwant := [][]int{{1, 2}, {3, 4}, {5}, {6}}
	if !reflect.DeepEqual(bgot, bwant) {
		t.Errorf("bundles: got %v, want %v", bgot, bwant)
	}
	tgot := quantizeTimes(handler.times(), b.DelayThreshold)
	twant := []int{0, 0, 0, 0}
	if !reflect.DeepEqual(tgot, twant) {
		t.Errorf("times: got %v, want %v", tgot, twant)
	}
}

func TestAddWait(t *testing.T) {
	var (
		mu     sync.Mutex
		events []string
	)
	event := func(s string) {
		mu.Lock()
		events = append(events, s)
		mu.Unlock()
	}

	handlec := make(chan int)
	done := make(chan struct{})
	b := NewBundler(int(0), func(interface{}) {
		<-handlec
		event("handle")
	})
	b.BufferedByteLimit = 3
	addw := func(sz int) {
		if err := b.AddWait(context.Background(), 0, sz); err != nil {
			t.Fatal(err)
		}
		event(fmt.Sprintf("addw(%d)", sz))
	}

	addw(2)
	go func() {
		addw(3) // blocks until first bundle is handled
		close(done)
	}()
	// Give addw(3) a chance to finish
	time.Sleep(100 * time.Millisecond)
	handlec <- 1 // handle the first bundle
	select {
	case <-time.After(time.Second):
		t.Fatal("timed out")
	case <-done:
	}
	want := []string{"addw(2)", "handle", "addw(3)"}
	if !reflect.DeepEqual(events, want) {
		t.Errorf("got  %v\nwant%v", events, want)
	}
}

func TestAddWaitCancel(t *testing.T) {
	b := NewBundler(int(0), func(interface{}) {})
	b.BufferedByteLimit = 3
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()
	err := b.AddWait(ctx, 0, 4)
	if want := context.Canceled; err != want {
		t.Fatalf("got %v, want %v", err, want)
	}
}

func TestBundlerErrors(t *testing.T) {
	// Use a handler that blocks forever, to force the bundler to run out of
	// memory.
	b := NewBundler(int(0), func(interface{}) { select {} })
	b.BundleByteLimit = 3
	b.BufferedByteLimit = 10

	if got, want := b.Add(1, 4), ErrOversizedItem; got != want {
		t.Fatalf("got %v, want %v", got, want)
	}

	for i := 0; i < 5; i++ {
		if err := b.Add(i, 2); err != nil {
			t.Fatal(err)
		}
	}
	if got, want := b.Add(5, 1), ErrOverflow; got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestModeError(t *testing.T) {
	// Call Add then AddWait.
	b := NewBundler(int(0), func(interface{}) {})
	b.BundleByteLimit = 4
	b.BufferedByteLimit = 4
	if err := b.Add(0, 2); err != nil {
		t.Fatal(err)
	}
	if got, want := b.AddWait(context.Background(), 0, 2), errMixedMethods; got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
	// Call AddWait then Add on new Bundler.
	b1 := NewBundler(int(0), func(interface{}) {})
	b1.BundleByteLimit = 4
	b1.BufferedByteLimit = 4
	if err := b1.AddWait(context.Background(), 0, 2); err != nil {
		t.Fatal(err)
	}
	if got, want := b1.Add(0, 2), errMixedMethods; got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// Check that no more than HandlerLimit handlers are active at once.
func TestConcurrentHandlersMax(t *testing.T) {
	const handlerLimit = 10
	var (
		mu          sync.Mutex
		active      int
		maxHandlers int
	)
	b := NewBundler(int(0), func(s interface{}) {
		mu.Lock()
		active++
		if active > maxHandlers {
			maxHandlers = active
		}
		if maxHandlers > handlerLimit {
			t.Errorf("too many handlers running (got %d; want %d)", maxHandlers, handlerLimit)
		}
		mu.Unlock()
		time.Sleep(1 * time.Millisecond) // let the scheduler work
		mu.Lock()
		active--
		mu.Unlock()
	})
	b.BundleCountThreshold = 5
	b.HandlerLimit = 10
	defer b.Flush()

	more := 0 // extra iterations past saturation
	for i := 0; more == 0 || i < more; i++ {
		mu.Lock()
		m := maxHandlers
		mu.Unlock()
		if m >= handlerLimit && more == 0 {
			// Run past saturation to check that we don't exceed the max.
			more = 2 * i
		}
		b.Add(i, 1)
	}
}

// Check that Flush doesn't return until all prior items have been handled.
func TestConcurrentFlush(t *testing.T) {
	var (
		mu    sync.Mutex
		items = make(map[int]bool)
	)
	b := NewBundler(int(0), func(s interface{}) {
		mu.Lock()
		for _, i := range s.([]int) {
			items[i] = true
		}
		mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	})
	b.BundleCountThreshold = 5
	b.HandlerLimit = 10
	defer b.Flush()

	var wg sync.WaitGroup
	defer wg.Wait()
	for i := 0; i < 50; i++ {
		b.Add(i, 1)
		if i%100 == 0 {
			i := i
			wg.Add(1)
			go func() {
				defer wg.Done()
				b.Flush()
				mu.Lock()
				defer mu.Unlock()
				for j := 0; j <= i; j++ {
					if !items[j] {
						// Cannot use Fatal, since we're in a non-test goroutine.
						t.Errorf("flush(%d): item %d not handled", i, j)
						break
					}
				}
			}()
		}
	}
}

type testHandler struct {
	mu sync.Mutex
	b  [][]int
	t  []time.Time
}

func (t *testHandler) bundles() [][]int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.b
}

func (t *testHandler) times() []time.Time {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.t
}

// Handler takes no time beyond adding to a list
func (t *testHandler) handleImmediate(b interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.b = append(t.b, b.([]int))
	t.t = append(t.t, time.Now())
}

// Handler takes 300 milliseconds
func (t *testHandler) handleSlow(b interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.b = append(t.b, b.([]int))
	t.t = append(t.t, time.Now())
	time.Sleep(300 * time.Millisecond)
}

// Handler takes one millisecond
func (t *testHandler) handleQuick(b interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.b = append(t.b, b.([]int))
	t.t = append(t.t, time.Now())
	time.Sleep(time.Millisecond)
}

// Round times to the nearest q and express them as the number of q
// since the first time.
// E.g. if q is 100ms, then a time within 50ms of the first time
// will be represented as 0, a time 150 to 250ms of the first time
// we be represented as 1, etc.
func quantizeTimes(times []time.Time, q time.Duration) []int {
	var rs []int
	for _, t := range times {
		d := t.Sub(times[0])
		r := int((d + q/2) / q)
		rs = append(rs, r)
	}
	return rs
}

func TestQuantizeTimes(t *testing.T) {
	quantum := 100 * time.Millisecond
	for _, test := range []struct {
		millis []int // times in milliseconds
		want   []int
	}{
		{[]int{10, 20, 30}, []int{0, 0, 0}},
		{[]int{0, 49, 50, 90}, []int{0, 0, 1, 1}},
		{[]int{0, 95, 170, 315}, []int{0, 1, 2, 3}},
	} {
		var times []time.Time
		for _, ms := range test.millis {
			times = append(times, time.Unix(0, int64(ms*1e6)))
		}
		got := quantizeTimes(times, quantum)
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("%v: got %v, want %v", test.millis, got, test.want)
		}
	}
}

// Measure the cost of adding a bunch of items only, though some handling may be
// happening in the background
func BenchmarkBundlerAdd(bench *testing.B) {
	// Unbundled case: one item per bundle.
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleImmediate)
	b.BundleCountThreshold = 1
	b.DelayThreshold = time.Second

	for i := 0; i < bench.N; i++ {
		if err := b.Add(i, 1); err != nil {
			bench.Fatal(err)
		}
	}
}

// Measure the cost of adding a bunch of items, and then waiting for them all to
// be handled, when handling is immediate (no delay)
func BenchmarkBundlerAddAndFlush(bench *testing.B) {
	// Unbundled case: one item per bundle.
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleImmediate)
	b.BundleCountThreshold = 1
	b.DelayThreshold = time.Second

	for i := 0; i < bench.N; i++ {
		if err := b.Add(i, 1); err != nil {
			bench.Fatal(err)
		}
	}
	b.Flush()
}

// Measure the cost of adding a bunch of items, and then waiting for them all to
// be handled, when handling a bundle (1 item only) takes one millisecond
func BenchmarkBundlerAddAndFlushSlow1(bench *testing.B) {
	// Unbundled case: one item per bundle.
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleQuick)
	b.BundleCountThreshold = 1
	b.DelayThreshold = time.Second

	for i := 0; i < bench.N; i++ {
		if err := b.Add(i, 1); err != nil {
			bench.Fatal(err)
		}
	}
	b.Flush()
}

// Measure the cost of adding a bunch of items, and then waiting for them all to
// be handled, when handling a bundle (25 items) takes one millisecond
func BenchmarkBundlerAddAndFlushSlow25(bench *testing.B) {
	// More realistic: 25 items per bundle
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handleQuick)
	b.BundleCountThreshold = 25
	b.DelayThreshold = time.Second

	for i := 0; i < bench.N; i++ {
		if err := b.Add(i, 1); err != nil {
			bench.Fatal(err)
		}
	}
	b.Flush()
}
