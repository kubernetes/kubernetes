// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package circonusgometrics

// A Gauge is an instantaneous measurement of a value.
//
// Use a gauge to track metrics which increase and decrease (e.g., amount of
// free memory).

import (
	"fmt"
)

// Gauge sets a gauge to a value
func (m *CirconusMetrics) Gauge(metric string, val interface{}) {
	m.SetGauge(metric, val)
}

// SetGauge sets a gauge to a value
func (m *CirconusMetrics) SetGauge(metric string, val interface{}) {
	m.gm.Lock()
	defer m.gm.Unlock()
	m.gauges[metric] = m.gaugeValString(val)
}

// RemoveGauge removes a gauge
func (m *CirconusMetrics) RemoveGauge(metric string) {
	m.gm.Lock()
	defer m.gm.Unlock()
	delete(m.gauges, metric)
}

// SetGaugeFunc sets a gauge to a function [called at flush interval]
func (m *CirconusMetrics) SetGaugeFunc(metric string, fn func() int64) {
	m.gfm.Lock()
	defer m.gfm.Unlock()
	m.gaugeFuncs[metric] = fn
}

// RemoveGaugeFunc removes a gauge function
func (m *CirconusMetrics) RemoveGaugeFunc(metric string) {
	m.gfm.Lock()
	defer m.gfm.Unlock()
	delete(m.gaugeFuncs, metric)
}

// gaugeValString converts an interface value (of a supported type) to a string
func (m *CirconusMetrics) gaugeValString(val interface{}) string {
	vs := ""
	switch v := val.(type) {
	default:
		// ignore it, unsupported type
	case int:
		vs = fmt.Sprintf("%d", v)
	case int8:
		vs = fmt.Sprintf("%d", v)
	case int16:
		vs = fmt.Sprintf("%d", v)
	case int32:
		vs = fmt.Sprintf("%d", v)
	case int64:
		vs = fmt.Sprintf("%d", v)
	case uint:
		vs = fmt.Sprintf("%d", v)
	case uint8:
		vs = fmt.Sprintf("%d", v)
	case uint16:
		vs = fmt.Sprintf("%d", v)
	case uint32:
		vs = fmt.Sprintf("%d", v)
	case uint64:
		vs = fmt.Sprintf("%d", v)
	case float32:
		vs = fmt.Sprintf("%f", v)
	case float64:
		vs = fmt.Sprintf("%f", v)
	}
	return vs
}
