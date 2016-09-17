// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package circonusgometrics

// A Text metric is an arbitrary string
//

// SetText sets a text metric
func (m *CirconusMetrics) SetText(metric string, val string) {
	m.SetTextValue(metric, val)
}

// SetTextValue sets a text metric
func (m *CirconusMetrics) SetTextValue(metric string, val string) {
	m.tm.Lock()
	defer m.tm.Unlock()
	m.text[metric] = val
}

// RemoveText removes a text metric
func (m *CirconusMetrics) RemoveText(metric string) {
	m.tm.Lock()
	defer m.tm.Unlock()
	delete(m.text, metric)
}

// SetTextFunc sets a text metric to a function [called at flush interval]
func (m *CirconusMetrics) SetTextFunc(metric string, fn func() string) {
	m.tfm.Lock()
	defer m.tfm.Unlock()
	m.textFuncs[metric] = fn
}

// RemoveTextFunc a text metric function
func (m *CirconusMetrics) RemoveTextFunc(metric string) {
	m.tfm.Lock()
	defer m.tfm.Unlock()
	delete(m.textFuncs, metric)
}
