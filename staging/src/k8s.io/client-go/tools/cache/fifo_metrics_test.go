/*
Copyright 2024 The Kubernetes Authors.

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
	"sync"
	"testing"
)

type testFifoMetric struct {
	inc int64
	dec int64

	lock sync.Mutex
}

func (m *testFifoMetric) Inc() {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.inc++
}

func (m *testFifoMetric) Dec() {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.dec++
}

func (m *testFifoMetric) cleanUp() {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.inc = 0
	m.dec = 0
}

func (m *testFifoMetric) countValue() int64 {
	m.lock.Lock()
	defer m.lock.Unlock()
	return m.inc
}

func (m *testFifoMetric) gaugeValue() int64 {
	m.lock.Lock()
	defer m.lock.Unlock()
	return m.inc - m.dec
}

type testFifoMetricsProvider struct {
	depth testFifoMetric
	adds  testFifoMetric
}

func (m *testFifoMetricsProvider) NewDepthMetric(name string) FifoGaugeMetric {
	return &m.depth
}

func (m *testFifoMetricsProvider) DeleteDepthMetric(name string) {
	m.depth.cleanUp()
}

func (m *testFifoMetricsProvider) NewAddsMetric(name string) CounterMetric {
	return &m.adds
}

func (m *testFifoMetricsProvider) DeleteAddsMetric(name string) {
	m.adds.cleanUp()
}

func TestFifoMetrics(t *testing.T) {
	mp := testFifoMetricsProvider{}
	SetProvider(&mp)
	f := NewDeltaFIFOWithOptions(DeltaFIFOOptions{KeyFunction: testFifoObjectKeyFunc, Name: "test-fifo"})
	const amount = 10000

	for i := 0; i < amount; i++ {
		f.Add(mkFifoObj(string([]rune{'a', rune(i)}), i+1))
	}
	for u := uint64(0); u < amount; u++ {
		f.Add(mkFifoObj(string([]rune{'b', rune(u)}), u+1))
	}

	if e, a := int64(20000), mp.adds.countValue(); e != a {
		t.Errorf("adds expected %v, got %v", e, a)
	}

	if e, a := int64(20000), mp.depth.gaugeValue(); e != a {
		t.Errorf("adds expected %v, got %v", e, a)
	}

	f.Pop(func(obj interface{}, isInInitialList bool) error {
		return nil
	})

	if e, a := int64(19999), mp.depth.gaugeValue(); e != a {
		t.Errorf("depth expected %v, got %v", e, a)
	}

	f.Close()

	if e, a := int64(0), mp.adds.countValue(); e != a {
		t.Errorf("adds expected %v, got %v", e, a)
	}

	if e, a := int64(0), mp.depth.gaugeValue(); e != a {
		t.Errorf("adds expected %v, got %v", e, a)
	}
}
