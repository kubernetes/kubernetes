// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package circonusgometrics

import (
	"github.com/circonus-labs/circonusllhist"
)

// Reset removes all existing counters and gauges.
func (m *CirconusMetrics) Reset() {
	m.cm.Lock()
	defer m.cm.Unlock()

	m.cfm.Lock()
	defer m.cfm.Unlock()

	m.gm.Lock()
	defer m.gm.Unlock()

	m.gfm.Lock()
	defer m.gfm.Unlock()

	m.hm.Lock()
	defer m.hm.Unlock()

	m.tm.Lock()
	defer m.tm.Unlock()

	m.tfm.Lock()
	defer m.tfm.Unlock()

	m.counters = make(map[string]uint64)
	m.counterFuncs = make(map[string]func() uint64)
	m.gauges = make(map[string]string)
	m.gaugeFuncs = make(map[string]func() int64)
	m.histograms = make(map[string]*Histogram)
	m.text = make(map[string]string)
	m.textFuncs = make(map[string]func() string)
}

// snapshot returns a copy of the values of all registered counters and gauges.
func (m *CirconusMetrics) snapshot() (c map[string]uint64, g map[string]string, h map[string]*circonusllhist.Histogram, t map[string]string) {
	m.cm.Lock()
	defer m.cm.Unlock()

	m.cfm.Lock()
	defer m.cfm.Unlock()

	m.gm.Lock()
	defer m.gm.Unlock()

	m.gfm.Lock()
	defer m.gfm.Unlock()

	m.hm.Lock()
	defer m.hm.Unlock()

	m.tm.Lock()
	defer m.tm.Unlock()

	m.tfm.Lock()
	defer m.tfm.Unlock()

	c = make(map[string]uint64, len(m.counters)+len(m.counterFuncs))
	for n, v := range m.counters {
		c[n] = v
	}

	for n, f := range m.counterFuncs {
		c[n] = f()
	}

	//g = make(map[string]int64, len(m.gauges)+len(m.gaugeFuncs))
	g = make(map[string]string, len(m.gauges)+len(m.gaugeFuncs))
	for n, v := range m.gauges {
		g[n] = v
	}

	for n, f := range m.gaugeFuncs {
		g[n] = m.gaugeValString(f())
	}

	h = make(map[string]*circonusllhist.Histogram, len(m.histograms))
	for n, hist := range m.histograms {
		h[n] = hist.hist.CopyAndReset()
	}

	t = make(map[string]string, len(m.text)+len(m.textFuncs))
	for n, v := range m.text {
		t[n] = v
	}

	for n, f := range m.textFuncs {
		t[n] = f()
	}

	return
}
