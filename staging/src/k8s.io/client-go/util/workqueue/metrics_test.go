/*
Copyright 2018 The Kubernetes Authors.

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

package workqueue

import (
	"sync"
	"testing"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

type testMetrics struct {
	added, gotten, finished int64

	updateCalled chan<- struct{}
}

func (m *testMetrics) add(item t)            { m.added++ }
func (m *testMetrics) get(item t)            { m.gotten++ }
func (m *testMetrics) done(item t)           { m.finished++ }
func (m *testMetrics) updateUnfinishedWork() { m.updateCalled <- struct{}{} }

func TestMetricShutdown(t *testing.T) {
	ch := make(chan struct{})
	m := &testMetrics{
		updateCalled: ch,
	}
	c := testingclock.NewFakeClock(time.Now())
	q := newQueue[any](c, DefaultQueue[any](), m, time.Millisecond)
	for !c.HasWaiters() {
		// Wait for the go routine to call NewTicker()
		time.Sleep(time.Millisecond)
	}

	c.Step(time.Millisecond)
	<-ch
	q.ShutDown()

	c.Step(time.Hour)
	select {
	default:
		return
	case <-ch:
		t.Errorf("Unexpected update after shutdown was called.")
	}
}

type testMetric struct {
	inc int64
	dec int64
	set float64

	observedValue float64
	observedCount int

	notifyCh chan<- struct{}

	lock sync.Mutex
}

func (m *testMetric) Inc() {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.inc++
	m.notify()
}

func (m *testMetric) Dec() {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.dec++
	m.notify()
}

func (m *testMetric) Set(f float64) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.set = f
	m.notify()
}

func (m *testMetric) Observe(f float64) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.observedValue = f
	m.observedCount++
	m.notify()
}

func (m *testMetric) gaugeValue() float64 {
	m.lock.Lock()
	defer m.lock.Unlock()
	if m.set != 0 {
		return m.set
	}
	return float64(m.inc - m.dec)
}

func (m *testMetric) observationValue() float64 {
	m.lock.Lock()
	defer m.lock.Unlock()
	return m.observedValue
}

func (m *testMetric) observationCount() int {
	m.lock.Lock()
	defer m.lock.Unlock()
	return m.observedCount
}

func (m *testMetric) notify() {
	if m.notifyCh != nil {
		m.notifyCh <- struct{}{}
	}
}

type testMetricsProvider struct {
	depth           testMetric
	adds            testMetric
	latency         testMetric
	duration        testMetric
	unfinished      testMetric
	longest         testMetric
	retries         testMetric
	waitingForDepth testMetric
}

func (m *testMetricsProvider) NewDepthMetric(name string) GaugeMetric {
	return &m.depth
}

func (m *testMetricsProvider) NewAddsMetric(name string) CounterMetric {
	return &m.adds
}

func (m *testMetricsProvider) NewLatencyMetric(name string) HistogramMetric {
	return &m.latency
}

func (m *testMetricsProvider) NewWorkDurationMetric(name string) HistogramMetric {
	return &m.duration
}

func (m *testMetricsProvider) NewUnfinishedWorkSecondsMetric(name string) SettableGaugeMetric {
	return &m.unfinished
}

func (m *testMetricsProvider) NewLongestRunningProcessorSecondsMetric(name string) SettableGaugeMetric {
	return &m.longest
}

func (m *testMetricsProvider) NewRetriesMetric(name string) CounterMetric {
	return &m.retries
}

func (m *testMetricsProvider) NewWaitingForQueueDepthMetric(name string) SettableGaugeMetric {
	return &m.waitingForDepth
}

func TestMetrics(t *testing.T) {
	mp := testMetricsProvider{}
	t0 := time.Unix(0, 0)
	c := testingclock.NewFakeClock(t0)
	config := QueueConfig{
		Name:            "test",
		Clock:           c,
		MetricsProvider: &mp,
	}
	q := newQueueWithConfig[any](config, time.Millisecond)
	defer q.ShutDown()
	for !c.HasWaiters() {
		// Wait for the go routine to call NewTicker()
		time.Sleep(time.Millisecond)
	}

	q.Add("foo")
	if e, a := 1.0, mp.adds.gaugeValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	if e, a := 1.0, mp.depth.gaugeValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	c.Step(50 * time.Microsecond)

	// Start processing
	i, _ := q.Get()
	if i != "foo" {
		t.Errorf("Expected %v, got %v", "foo", i)
	}

	if e, a := 5e-05, mp.latency.observationValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 1, mp.latency.observationCount(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// Add it back while processing; multiple adds of the same item are
	// de-duped.
	q.Add(i)
	q.Add(i)
	q.Add(i)
	q.Add(i)
	q.Add(i)
	if e, a := 2.0, mp.adds.gaugeValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	// One thing remains in the queue
	if e, a := 1.0, mp.depth.gaugeValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	c.Step(25 * time.Microsecond)

	// Finish it up
	q.Done(i)

	if e, a := 2.5e-05, mp.duration.observationValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 1, mp.duration.observationCount(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// One thing remains in the queue
	if e, a := 1.0, mp.depth.gaugeValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// It should be back on the queue
	i, _ = q.Get()
	if i != "foo" {
		t.Errorf("Expected %v, got %v", "foo", i)
	}

	if e, a := 2.5e-05, mp.latency.observationValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2, mp.latency.observationCount(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// use a channel to ensure we don't look at the metric before it's
	// been set.
	ch := make(chan struct{}, 1)
	longestCh := make(chan struct{}, 1)
	mp.unfinished.notifyCh = ch
	mp.longest.notifyCh = longestCh
	c.Step(time.Millisecond)
	<-ch
	mp.unfinished.notifyCh = nil
	if e, a := .001, mp.unfinished.gaugeValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	<-longestCh
	mp.longest.notifyCh = nil
	if e, a := .001, mp.longest.gaugeValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// Finish that one up
	q.Done(i)
	if e, a := .001, mp.duration.observationValue(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2, mp.duration.observationCount(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}
