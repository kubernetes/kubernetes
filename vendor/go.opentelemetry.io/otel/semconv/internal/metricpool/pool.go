// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package metricpool provides shared pools for semantic convention metric
// measurement options.
package metricpool

import (
	"sync"

	"go.opentelemetry.io/otel/metric"
)

var (
	addOptionsPool = sync.Pool{New: func() any {
		o := make([]metric.AddOption, 0, 1)
		return &o
	}}
	recOptionsPool = sync.Pool{New: func() any {
		o := make([]metric.RecordOption, 0, 1)
		return &o
	}}
)

// AddOptions returns a pooled AddOption slice.
func AddOptions() *[]metric.AddOption {
	return addOptionsPool.Get().(*[]metric.AddOption)
}

// PutAddOptions clears, resets, and returns o to the shared AddOption pool.
func PutAddOptions(o *[]metric.AddOption) {
	clear(*o)
	*o = (*o)[:0]
	addOptionsPool.Put(o)
}

// RecordOptions returns a pooled RecordOption slice.
func RecordOptions() *[]metric.RecordOption {
	return recOptionsPool.Get().(*[]metric.RecordOption)
}

// PutRecordOptions clears, resets, and returns o to the shared RecordOption
// pool.
func PutRecordOptions(o *[]metric.RecordOption) {
	clear(*o)
	*o = (*o)[:0]
	recOptionsPool.Put(o)
}
