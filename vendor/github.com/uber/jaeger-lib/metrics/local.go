// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package metrics

import (
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/codahale/hdrhistogram"
)

// This is intentionally very similar to github.com/codahale/metrics, the
// main difference being that counters/gauges are scoped to the provider
// rather than being global (to facilitate testing).

// A LocalBackend is a metrics provider which aggregates data in-vm, and
// allows exporting snapshots to shove the data into a remote collector
type LocalBackend struct {
	cm       sync.Mutex
	gm       sync.Mutex
	tm       sync.Mutex
	counters map[string]*int64
	gauges   map[string]*int64
	timers   map[string]*localBackendTimer
	stop     chan struct{}
	wg       sync.WaitGroup
	TagsSep  string
	TagKVSep string
}

// NewLocalBackend returns a new LocalBackend. The collectionInterval is the histogram
// time window for each timer.
func NewLocalBackend(collectionInterval time.Duration) *LocalBackend {
	b := &LocalBackend{
		counters: make(map[string]*int64),
		gauges:   make(map[string]*int64),
		timers:   make(map[string]*localBackendTimer),
		stop:     make(chan struct{}),
		TagsSep:  "|",
		TagKVSep: "=",
	}
	if collectionInterval == 0 {
		// Use one histogram time window for all timers
		return b
	}
	b.wg.Add(1)
	go b.runLoop(collectionInterval)
	return b
}

// Clear discards accumulated stats
func (b *LocalBackend) Clear() {
	b.cm.Lock()
	defer b.cm.Unlock()
	b.gm.Lock()
	defer b.gm.Unlock()
	b.tm.Lock()
	defer b.tm.Unlock()
	b.counters = make(map[string]*int64)
	b.gauges = make(map[string]*int64)
	b.timers = make(map[string]*localBackendTimer)
}

func (b *LocalBackend) runLoop(collectionInterval time.Duration) {
	defer b.wg.Done()
	ticker := time.NewTicker(collectionInterval)
	for {
		select {
		case <-ticker.C:
			b.tm.Lock()
			timers := make(map[string]*localBackendTimer, len(b.timers))
			for timerName, timer := range b.timers {
				timers[timerName] = timer
			}
			b.tm.Unlock()

			for _, t := range timers {
				t.Lock()
				t.hist.Rotate()
				t.Unlock()
			}
		case <-b.stop:
			ticker.Stop()
			return
		}
	}
}

// IncCounter increments a counter value
func (b *LocalBackend) IncCounter(name string, tags map[string]string, delta int64) {
	name = GetKey(name, tags, b.TagsSep, b.TagKVSep)
	b.cm.Lock()
	defer b.cm.Unlock()
	counter := b.counters[name]
	if counter == nil {
		b.counters[name] = new(int64)
		*b.counters[name] = delta
		return
	}
	atomic.AddInt64(counter, delta)
}

// UpdateGauge updates the value of a gauge
func (b *LocalBackend) UpdateGauge(name string, tags map[string]string, value int64) {
	name = GetKey(name, tags, b.TagsSep, b.TagKVSep)
	b.gm.Lock()
	defer b.gm.Unlock()
	gauge := b.gauges[name]
	if gauge == nil {
		b.gauges[name] = new(int64)
		*b.gauges[name] = value
		return
	}
	atomic.StoreInt64(gauge, value)
}

// RecordTimer records a timing duration
func (b *LocalBackend) RecordTimer(name string, tags map[string]string, d time.Duration) {
	name = GetKey(name, tags, b.TagsSep, b.TagKVSep)
	timer := b.findOrCreateTimer(name)
	timer.Lock()
	timer.hist.Current.RecordValue(int64(d / time.Millisecond))
	timer.Unlock()
}

func (b *LocalBackend) findOrCreateTimer(name string) *localBackendTimer {
	b.tm.Lock()
	defer b.tm.Unlock()
	if t, ok := b.timers[name]; ok {
		return t
	}

	t := &localBackendTimer{
		hist: hdrhistogram.NewWindowed(5, 0, int64((5*time.Minute)/time.Millisecond), 1),
	}
	b.timers[name] = t
	return t
}

type localBackendTimer struct {
	sync.Mutex
	hist *hdrhistogram.WindowedHistogram
}

var (
	percentiles = map[string]float64{
		"P50":  50,
		"P75":  75,
		"P90":  90,
		"P95":  95,
		"P99":  99,
		"P999": 99.9,
	}
)

// Snapshot captures a snapshot of the current counter and gauge values
func (b *LocalBackend) Snapshot() (counters, gauges map[string]int64) {
	b.cm.Lock()
	defer b.cm.Unlock()

	counters = make(map[string]int64, len(b.counters))
	for name, value := range b.counters {
		counters[name] = atomic.LoadInt64(value)
	}

	b.gm.Lock()
	defer b.gm.Unlock()

	gauges = make(map[string]int64, len(b.gauges))
	for name, value := range b.gauges {
		gauges[name] = atomic.LoadInt64(value)
	}

	b.tm.Lock()
	timers := make(map[string]*localBackendTimer)
	for timerName, timer := range b.timers {
		timers[timerName] = timer
	}
	b.tm.Unlock()

	for timerName, timer := range timers {
		timer.Lock()
		hist := timer.hist.Merge()
		timer.Unlock()
		for name, q := range percentiles {
			gauges[timerName+"."+name] = hist.ValueAtQuantile(q)
		}
	}

	return
}

// Stop cleanly closes the background goroutine spawned by NewLocalBackend.
func (b *LocalBackend) Stop() {
	close(b.stop)
	b.wg.Wait()
}

// GetKey converts name+tags into a single string of the form
// "name|tag1=value1|...|tagN=valueN", where tag names are
// sorted alphabetically.
func GetKey(name string, tags map[string]string, tagsSep string, tagKVSep string) string {
	keys := make([]string, 0, len(tags))
	for k := range tags {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	key := name
	for _, k := range keys {
		key = key + tagsSep + k + tagKVSep + tags[k]
	}
	return key
}

type stats struct {
	name         string
	tags         map[string]string
	localBackend *LocalBackend
}

type localTimer struct {
	stats
}

func (l *localTimer) Record(d time.Duration) {
	l.localBackend.RecordTimer(l.name, l.tags, d)
}

type localCounter struct {
	stats
}

func (l *localCounter) Inc(delta int64) {
	l.localBackend.IncCounter(l.name, l.tags, delta)
}

type localGauge struct {
	stats
}

func (l *localGauge) Update(value int64) {
	l.localBackend.UpdateGauge(l.name, l.tags, value)
}

// LocalFactory stats factory that creates metrics that are stored locally
type LocalFactory struct {
	*LocalBackend
	namespace string
	tags      map[string]string
}

// NewLocalFactory returns a new LocalMetricsFactory
func NewLocalFactory(collectionInterval time.Duration) *LocalFactory {
	return &LocalFactory{
		LocalBackend: NewLocalBackend(collectionInterval),
	}
}

// appendTags adds the tags to the namespace tags and returns a combined map.
func (l *LocalFactory) appendTags(tags map[string]string) map[string]string {
	newTags := make(map[string]string)
	for k, v := range l.tags {
		newTags[k] = v
	}
	for k, v := range tags {
		newTags[k] = v
	}
	return newTags
}

func (l *LocalFactory) newNamespace(name string) string {
	if l.namespace == "" {
		return name
	}
	return l.namespace + "." + name
}

// Counter returns a local stats counter
func (l *LocalFactory) Counter(name string, tags map[string]string) Counter {
	return &localCounter{
		stats{
			name:         l.newNamespace(name),
			tags:         l.appendTags(tags),
			localBackend: l.LocalBackend,
		},
	}
}

// Timer returns a local stats timer.
func (l *LocalFactory) Timer(name string, tags map[string]string) Timer {
	return &localTimer{
		stats{
			name:         l.newNamespace(name),
			tags:         l.appendTags(tags),
			localBackend: l.LocalBackend,
		},
	}
}

// Gauge returns a local stats gauge.
func (l *LocalFactory) Gauge(name string, tags map[string]string) Gauge {
	return &localGauge{
		stats{
			name:         l.newNamespace(name),
			tags:         l.appendTags(tags),
			localBackend: l.LocalBackend,
		},
	}
}

// Namespace returns a new namespace.
func (l *LocalFactory) Namespace(name string, tags map[string]string) Factory {
	return &LocalFactory{
		namespace:    l.newNamespace(name),
		tags:         l.appendTags(tags),
		LocalBackend: l.LocalBackend,
	}
}
