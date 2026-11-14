//go:build !windows

/*
Copyright The Kubernetes Authors.

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

package cacher

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	cachertesting "k8s.io/apiserver/pkg/storage/cacher/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// BenchmarkWatchFairness measures what slow readers cost the readers beside
// them, and how well the slow readers themselves are served. It sees only what
// clients see. It needs wall-clock time, and reports one run; run it with:
//
//	go test ./staging/src/k8s.io/apiserver/pkg/storage/cacher/ -run xxx -bench BenchmarkWatchFairness -benchtime 1x
//
// The tax is how far the healthy-* metrics are from those of slow=none at the
// same rate and with the same healthy readers. Ideally it is nothing, whatever
// n is, and nobody is closed.
//
//	healthy-late-%    time that readers waited for an event that was due over 5 ms ago
//	healthy-stale-ms  oldest event that a reader lacked, counted from when it was due
//	healthy-closes    watches closed by the server
//	slow-late-%       as healthy-late-%: a slow reader that isn't reading isn't waiting
//	slow-closes       as healthy-closes
//	*-expired         watches that were too far behind to go on, if any
//	open-ms           mean time to open one more watch
//	cpu-%             of one CPU, readers included, so compare rows of the same n
//
// An event is due when the schedule says it is written, so a cacher that
// blocks the writer pays for that too. Clients whose watch is closed open it
// again at once. That costs them little here, and an API server much more.
func BenchmarkWatchFairness(b *testing.B) {
	steady := reader{name: "steady"}
	// A little longer than its channels last: too short to be closed for, if nobody else is slow.
	hiccup := reader{name: "hiccup", pause: 206 * time.Millisecond, pauseEvery: 2 * time.Second, pausesApart: true}
	hiccupLong := reader{name: "hiccup-long", pause: 300 * time.Millisecond, pauseEvery: 2 * time.Second, pausesApart: true}

	none := reader{name: "none"}
	stuck := reader{name: "stuck", abandonsAfter: time.Second}
	trickle := reader{name: "trickle", eventsPerSecond: 100}
	pause := reader{name: "pause", eventsPerSecond: 5000, pause: time.Second, pauseEvery: 2 * time.Second}
	half := reader{name: "half", eventsPerSecond: 1500}

	for _, scenario := range []struct {
		eventsPerSecond int
		healthy, slow   reader
		counts          []int
		shortHistory    bool
	}{
		{eventsPerSecond: 1000, healthy: steady, slow: none, counts: []int{0}},
		{eventsPerSecond: 1000, healthy: steady, slow: stuck, counts: []int{1, 100, 1000}},
		{eventsPerSecond: 1000, healthy: steady, slow: trickle, counts: []int{1, 100, 1000}},
		{eventsPerSecond: 1000, healthy: steady, slow: pause, counts: []int{1, 100, 1000}},
		{eventsPerSecond: 3000, healthy: steady, slow: none, counts: []int{0}},
		{eventsPerSecond: 3000, healthy: steady, slow: half, counts: []int{1}},
		{eventsPerSecond: 1000, healthy: hiccup, slow: none, counts: []int{0}},
		{eventsPerSecond: 1000, healthy: hiccup, slow: stuck, counts: []int{1, 100}},
		{eventsPerSecond: 1000, healthy: hiccup, slow: trickle, counts: []int{1, 100}},
		{eventsPerSecond: 1000, healthy: hiccup, slow: pause, counts: []int{1, 100}},
		{eventsPerSecond: 1000, healthy: hiccupLong, slow: none, counts: []int{0}},
		{eventsPerSecond: 1000, healthy: hiccupLong, slow: trickle, counts: []int{1, 100}},
		// Watches opened while the history is short, as after a restart, have the smallest channels.
		{eventsPerSecond: 1000, healthy: hiccup, slow: none, counts: []int{0}, shortHistory: true},
	} {
		for _, count := range scenario.counts {
			for _, stallResume := range []bool{false, true} {
				b.Run(fmt.Sprintf("rate=%d/healthy=%s/slow=%s/n=%d/shortHistory=%v/StallResume=%v", scenario.eventsPerSecond, scenario.healthy.name, scenario.slow.name, count, scenario.shortHistory, stallResume), func(b *testing.B) {
					featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.WatchCacheStallResume, stallResume)
					for b.Loop() {
						f := newWatchFairness(b, scenario.eventsPerSecond, scenario.shortHistory)
						healthy := f.addReaders(scenario.healthy, watchFairnessHealthyReaders)
						slow := f.addReaders(scenario.slow, count)
						f.run()
						f.report("healthy", healthy)
						if count > 0 {
							f.report("slow", slow)
						}
					}
					b.ReportMetric(0, "ns/op")
				})
			}
		}
	}
}

const (
	watchFairnessDuration       = 5 * time.Second
	watchFairnessHealthyReaders = 10
	watchFairnessLate           = 5 * time.Millisecond
	// watchFairnessHistory is long enough for watches to get their full channels.
	watchFairnessHistory = 20000
	// watchFairnessOpenEach is out of step with whatever the cacher does periodically.
	watchFairnessOpenEach = 153 * time.Millisecond
	watchFairnessPodBytes = 2000
)

// reader says how a client reads. The zero value reads as fast as it can.
type reader struct {
	name              string
	abandonsAfter     time.Duration // reads nothing, and opens a new watch after that
	eventsPerSecond   int
	pause, pauseEvery time.Duration
	pausesApart       bool // rather than all readers at once
}

type watchFairness struct {
	b               *testing.B
	cacher          *Cacher
	eventsPerSecond int
	readers         []*watchReader
	written         atomic.Uint64 // the latest resourceVersion
	firstRV         uint64        // the first that is on the schedule
	start           time.Time
	ctx             context.Context // of all watches, until the end of the run
}

func newWatchFairness(b *testing.B, eventsPerSecond int, shortHistory bool) *watchFairness {
	cacher, _, err := newTestCacher(&cachertesting.MockStorage{})
	if err != nil {
		b.Fatal(err)
	}
	f := &watchFairness{b: b, cacher: cacher, eventsPerSecond: eventsPerSecond}
	f.written.Store(cacher.watchCache.getListResourceVersion())
	f.write()
	for !shortHistory && f.written.Load() < watchFairnessHistory {
		f.write()
	}
	f.firstRV = f.written.Load() + 1
	return f
}

// write adds the event whose resourceVersion is one more than the last.
func (f *watchFairness) write() {
	rv := f.written.Load() + 1
	if err := f.cacher.watchCache.Update(&example.Pod{
		Name:            fmt.Sprintf("pod-%d", rv%2000),
		Namespace:       "ns",
		ResourceVersion: strconv.FormatUint(rv, 10),
		Annotations:     map[string]string{"padding": strings.Repeat("x", watchFairnessPodBytes)},
	}); err != nil {
		panic(err)
	}
	f.written.Store(rv)
}

func (f *watchFairness) due(rv uint64) time.Duration {
	return time.Duration(rv-f.firstRV) * time.Second / time.Duration(f.eventsPerSecond)
}

// now stops at the end of the run: the time that it takes to stop doesn't count.
func (f *watchFairness) now() time.Duration {
	return min(time.Since(f.start), watchFairnessDuration)
}

// open gives every watch its own context, as requests have: the channel of a
// shared one would be contended.
func (f *watchFairness) open(rv uint64) (w watch.Interface, stop func()) {
	ctx, cancel := context.WithCancel(f.ctx)
	predicate := storage.Everything
	predicate.AllowWatchBookmarks = true
	w, err := f.cacher.Watch(ctx, "/pods/", storage.ListOptions{
		ResourceVersion: strconv.FormatUint(rv, 10),
		Predicate:       predicate,
		Recursive:       true,
	})
	if err != nil {
		panic(err) // Watch reports errors as events once the cacher is ready.
	}
	return w, func() { w.Stop(); cancel() }
}

func (f *watchFairness) addReaders(config reader, count int) []*watchReader {
	readers := make([]*watchReader, count)
	for i := range readers {
		r := &watchReader{reader: config, f: f, seen: f.written.Load()}
		r.nextPause = r.pauseEvery / 2
		if r.pausesApart {
			r.nextPause += r.pauseEvery * time.Duration(i) / time.Duration(count)
		}
		readers[i] = r
	}
	f.readers = append(f.readers, readers...)
	return readers
}

func (f *watchFairness) run() {
	defer f.cacher.Stop()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f.ctx = ctx
	var opening, running sync.WaitGroup
	for _, r := range f.readers {
		opening.Go(r.open)
	}
	opening.Wait()

	cpuBefore := cpuTime()
	f.start = time.Now()
	for _, r := range f.readers {
		running.Go(r.run)
	}
	running.Go(f.measureOpen)
	for f.now() < watchFairnessDuration {
		f.write()
		time.Sleep(f.due(f.written.Load()+1) - f.now())
	}
	cpu, took := cpuTime()-cpuBefore, time.Since(f.start)
	cancel()
	running.Wait()
	f.b.ReportMetric(100*cpu.Seconds()/took.Seconds(), "cpu-%")
}

func (f *watchFairness) measureOpen() {
	var opens int
	var took time.Duration
	for next := watchFairnessOpenEach; f.now() < watchFairnessDuration; next += watchFairnessOpenEach {
		began := f.now()
		_, stop := f.open(f.written.Load())
		took += f.now() - began
		opens++
		stop()
		time.Sleep(next - f.now())
	}
	f.b.ReportMetric(float64(took.Microseconds())/1000/float64(opens), "open-ms")
}

func (f *watchFairness) report(name string, readers []*watchReader) {
	var late, stale time.Duration
	var closes, expired int
	for _, r := range readers {
		late += r.late
		stale = max(stale, r.stale)
		closes += r.closes
		expired += r.expired
	}
	f.b.ReportMetric(100*late.Seconds()/(watchFairnessDuration.Seconds()*float64(len(readers))), name+"-late-%")
	if name == "healthy" {
		f.b.ReportMetric(float64(stale.Milliseconds()), name+"-stale-ms")
	}
	f.b.ReportMetric(float64(closes), name+"-closes")
	if expired > 0 {
		f.b.ReportMetric(float64(expired), name+"-expired")
	}
}

type watchReader struct {
	reader
	f         *watchFairness
	watch     watch.Interface
	stop      func()
	seen      uint64        // the resourceVersion of the last event
	waitsFrom time.Duration // when it asked for the next
	nextPause time.Duration
	nextRead  time.Time // at eventsPerSecond

	late, stale time.Duration
	closes      int
	expired     int
}

func (r *watchReader) open() {
	r.watch, r.stop = r.f.open(r.seen)
}

func (r *watchReader) run() {
	for r.f.ctx.Err() == nil {
		if r.abandonsAfter > 0 {
			r.abandon()
		} else {
			r.readAll()
		}
		r.stop()
		r.open()
	}
	r.stop()
	r.account() // the event that it still lacks
}

// abandon keeps the watch for a while and then looks if the server closed it.
func (r *watchReader) abandon() {
	defer r.startFromLatest()
	select {
	case <-r.f.ctx.Done():
		return
	case <-time.After(r.abandonsAfter):
	}
	for {
		select {
		case _, ok := <-r.watch.ResultChan():
			if ok {
				continue
			}
			r.closes++
		default:
		}
		return
	}
}

// readAll reads until the watch is closed.
func (r *watchReader) readAll() {
	for {
		event, ok := <-r.watch.ResultChan()
		if r.f.ctx.Err() != nil {
			return
		}
		if !ok {
			r.closes++
			return
		}
		r.handle(event)
	}
}

// startFromLatest is what a client that is too far behind does.
func (r *watchReader) startFromLatest() {
	r.seen = r.f.written.Load()
	r.waitsFrom = r.f.now()
}

func (r *watchReader) handle(event watch.Event) {
	if event.Type == watch.Error {
		if err := errors.FromObject(event.Object); !errors.IsResourceExpired(err) {
			panic(err)
		}
		r.account()
		r.expired++
		r.startFromLatest()
		return
	}
	rv, err := r.f.cacher.versioner.ObjectResourceVersion(event.Object)
	if err != nil {
		panic(err)
	}
	if event.Type == watch.Bookmark {
		if rv > r.seen {
			r.f.b.Errorf("bookmark %d came after event %d", rv, r.seen)
		}
		return
	}
	if rv != r.seen+1 {
		r.f.b.Errorf("event %d came after event %d", rv, r.seen)
	}
	r.account()
	r.seen = rv
	if err := cachertesting.Codecs.LegacyCodec(examplev1.SchemeGroupVersion).Encode(event.Object, io.Discard); err != nil {
		panic(err)
	}
	if r.pause > 0 && r.f.now() > r.nextPause {
		time.Sleep(r.pause)
		r.nextPause += r.pauseEvery
	}
	if r.eventsPerSecond > 0 {
		r.keepPace()
	}
	r.waitsFrom = r.f.now()
}

// account adds the wait for the event that follows the last.
func (r *watchReader) account() {
	now, due := r.f.now(), r.f.due(r.seen+1)
	r.stale = max(r.stale, now-due)
	r.late += max(0, now-max(due+watchFairnessLate, r.waitsFrom))
}

func (r *watchReader) keepPace() {
	if time.Since(r.nextRead) > 5*time.Millisecond {
		r.nextRead = time.Now() // It can't make up for the time that it waited.
	}
	r.nextRead = r.nextRead.Add(time.Second / time.Duration(r.eventsPerSecond))
	if ahead := time.Until(r.nextRead); ahead > 2*time.Millisecond {
		time.Sleep(ahead)
	}
}

// cpuTime is the CPU time used by this process.
func cpuTime() time.Duration {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		panic(err)
	}
	return time.Duration(usage.Utime.Nano() + usage.Stime.Nano())
}
