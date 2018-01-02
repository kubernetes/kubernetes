// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bundler

import (
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestBundlerCount1(t *testing.T) {
	// Unbundled case: one item per bundle.
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handle)
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
	b := NewBundler(int(0), handler.handle)
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

func TestBundlerByteThreshold(t *testing.T) {
	handler := &testHandler{}
	b := NewBundler(int(0), handler.handle)
	b.BundleCountThreshold = 10
	b.BundleByteThreshold = 3
	add := func(i interface{}, s int) {
		if err := b.Add(i, s); err != nil {
			t.Fatal(err)
		}
	}

	add(1, 1)
	add(2, 2)
	// Hit byte threshold: bundle = 1, 2
	add(3, 1)
	add(4, 1)
	add(5, 2)
	// Passed byte threshold, but not limit: bundle = 3, 4, 5
	add(6, 1)
	b.Flush()
	bgot := handler.bundles()
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
	b := NewBundler(int(0), handler.handle)
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

func (t *testHandler) handle(b interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.b = append(t.b, b.([]int))
	t.t = append(t.t, time.Now())
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
