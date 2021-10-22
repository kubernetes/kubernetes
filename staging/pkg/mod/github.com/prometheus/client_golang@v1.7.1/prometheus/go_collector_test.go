// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus

import (
	"runtime"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
)

func TestGoCollectorGoroutines(t *testing.T) {
	var (
		c               = NewGoCollector()
		metricCh        = make(chan Metric)
		waitCh          = make(chan struct{})
		endGoroutineCh  = make(chan struct{})
		endCollectionCh = make(chan struct{})
		old             = -1
	)
	defer func() {
		close(endGoroutineCh)
		// Drain the collect channel to prevent goroutine leak.
		for {
			select {
			case <-metricCh:
			case <-endCollectionCh:
				return
			}
		}
	}()

	go func() {
		c.Collect(metricCh)
		for i := 1; i <= 10; i++ {
			// Start 10 goroutines to be sure we'll detect an
			// increase even if unrelated goroutines happen to
			// terminate during this test.
			go func(c <-chan struct{}) {
				<-c
			}(endGoroutineCh)
		}
		<-waitCh
		c.Collect(metricCh)
		close(endCollectionCh)
	}()

	for {
		select {
		case m := <-metricCh:
			// m can be Gauge or Counter,
			// currently just test the go_goroutines Gauge
			// and ignore others.
			if m.Desc().fqName != "go_goroutines" {
				continue
			}
			pb := &dto.Metric{}
			m.Write(pb)
			if pb.GetGauge() == nil {
				continue
			}

			if old == -1 {
				old = int(pb.GetGauge().GetValue())
				close(waitCh)
				continue
			}

			if diff := old - int(pb.GetGauge().GetValue()); diff > -1 {
				t.Errorf("want at least one new goroutine, got %d fewer", diff)
			}
		case <-time.After(1 * time.Second):
			t.Fatalf("expected collect timed out")
		}
		break
	}
}

func TestGoCollectorGC(t *testing.T) {
	var (
		c               = NewGoCollector()
		metricCh        = make(chan Metric)
		waitCh          = make(chan struct{})
		endCollectionCh = make(chan struct{})
		oldGC           uint64
		oldPause        float64
	)

	go func() {
		c.Collect(metricCh)
		// force GC
		runtime.GC()
		<-waitCh
		c.Collect(metricCh)
		close(endCollectionCh)
	}()

	defer func() {
		// Drain the collect channel to prevent goroutine leak.
		for {
			select {
			case <-metricCh:
			case <-endCollectionCh:
				return
			}
		}
	}()

	first := true
	for {
		select {
		case metric := <-metricCh:
			pb := &dto.Metric{}
			metric.Write(pb)
			if pb.GetSummary() == nil {
				continue
			}
			if len(pb.GetSummary().Quantile) != 5 {
				t.Errorf("expected 4 buckets, got %d", len(pb.GetSummary().Quantile))
			}
			for idx, want := range []float64{0.0, 0.25, 0.5, 0.75, 1.0} {
				if *pb.GetSummary().Quantile[idx].Quantile != want {
					t.Errorf("bucket #%d is off, got %f, want %f", idx, *pb.GetSummary().Quantile[idx].Quantile, want)
				}
			}
			if first {
				first = false
				oldGC = *pb.GetSummary().SampleCount
				oldPause = *pb.GetSummary().SampleSum
				close(waitCh)
				continue
			}
			if diff := *pb.GetSummary().SampleCount - oldGC; diff < 1 {
				t.Errorf("want at least 1 new garbage collection run, got %d", diff)
			}
			if diff := *pb.GetSummary().SampleSum - oldPause; diff <= 0 {
				t.Errorf("want an increase in pause time, got a change of %f", diff)
			}
		case <-time.After(1 * time.Second):
			t.Fatalf("expected collect timed out")
		}
		break
	}
}

func TestGoCollectorMemStats(t *testing.T) {
	var (
		c   = NewGoCollector().(*goCollector)
		got uint64
	)

	checkCollect := func(want uint64) {
		metricCh := make(chan Metric)
		endCh := make(chan struct{})

		go func() {
			c.Collect(metricCh)
			close(endCh)
		}()
	Collect:
		for {
			select {
			case metric := <-metricCh:
				if metric.Desc().fqName != "go_memstats_alloc_bytes" {
					continue Collect
				}
				pb := &dto.Metric{}
				metric.Write(pb)
				got = uint64(pb.GetGauge().GetValue())
			case <-endCh:
				break Collect
			}
		}
		if want != got {
			t.Errorf("unexpected value of go_memstats_alloc_bytes, want %d, got %d", want, got)
		}
	}

	// Speed up the timing to make the test faster.
	c.msMaxWait = 5 * time.Millisecond
	c.msMaxAge = 50 * time.Millisecond

	// Scenario 1: msRead responds slowly, no previous memstats available,
	// msRead is executed anyway.
	c.msRead = func(ms *runtime.MemStats) {
		time.Sleep(20 * time.Millisecond)
		ms.Alloc = 1
	}
	checkCollect(1)
	// Now msLast is set.
	c.msMtx.Lock()
	if want, got := uint64(1), c.msLast.Alloc; want != got {
		t.Errorf("unexpected of msLast.Alloc, want %d, got %d", want, got)
	}
	c.msMtx.Unlock()

	// Scenario 2: msRead responds fast, previous memstats available, new
	// value collected.
	c.msRead = func(ms *runtime.MemStats) {
		ms.Alloc = 2
	}
	checkCollect(2)
	// msLast is set, too.
	c.msMtx.Lock()
	if want, got := uint64(2), c.msLast.Alloc; want != got {
		t.Errorf("unexpected of msLast.Alloc, want %d, got %d", want, got)
	}
	c.msMtx.Unlock()

	// Scenario 3: msRead responds slowly, previous memstats available, old
	// value collected.
	c.msRead = func(ms *runtime.MemStats) {
		time.Sleep(20 * time.Millisecond)
		ms.Alloc = 3
	}
	checkCollect(2)
	// After waiting, new value is still set in msLast.
	time.Sleep(80 * time.Millisecond)
	c.msMtx.Lock()
	if want, got := uint64(3), c.msLast.Alloc; want != got {
		t.Errorf("unexpected of msLast.Alloc, want %d, got %d", want, got)
	}
	c.msMtx.Unlock()

	// Scenario 4: msRead responds slowly, previous memstats is too old, new
	// value collected.
	c.msRead = func(ms *runtime.MemStats) {
		time.Sleep(20 * time.Millisecond)
		ms.Alloc = 4
	}
	checkCollect(4)
	c.msMtx.Lock()
	if want, got := uint64(4), c.msLast.Alloc; want != got {
		t.Errorf("unexpected of msLast.Alloc, want %d, got %d", want, got)
	}
	c.msMtx.Unlock()
}
