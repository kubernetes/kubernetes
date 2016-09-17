// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package circonusgometrics

// A Counter is a monotonically increasing unsigned integer.
//
// Use a counter to derive rates (e.g., record total number of requests, derive
// requests per second).

// Increment counter by 1
func (m *CirconusMetrics) Increment(metric string) {
	m.Add(metric, 1)
}

// IncrementByValue updates counter by supplied value
func (m *CirconusMetrics) IncrementByValue(metric string, val uint64) {
	m.Add(metric, val)
}

// Set a counter to specific value
func (m *CirconusMetrics) Set(metric string, val uint64) {
	m.cm.Lock()
	defer m.cm.Unlock()
	m.counters[metric] = val
}

// Add updates counter by supplied value
func (m *CirconusMetrics) Add(metric string, val uint64) {
	m.cm.Lock()
	defer m.cm.Unlock()
	m.counters[metric] += val
}

// RemoveCounter removes the named counter
func (m *CirconusMetrics) RemoveCounter(metric string) {
	m.cm.Lock()
	defer m.cm.Unlock()
	delete(m.counters, metric)
}

// SetCounterFunc set counter to a function [called at flush interval]
func (m *CirconusMetrics) SetCounterFunc(metric string, fn func() uint64) {
	m.cfm.Lock()
	defer m.cfm.Unlock()
	m.counterFuncs[metric] = fn
}

// RemoveCounterFunc removes the named counter function
func (m *CirconusMetrics) RemoveCounterFunc(metric string) {
	m.cfm.Lock()
	defer m.cfm.Unlock()
	delete(m.counterFuncs, metric)
}
