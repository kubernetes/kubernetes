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

// Package internal holds shared state for the metrics package that must not be
// directly accessible by external callers.
package internal

import "sync/atomic"

var nativeHistogramsEnabled atomic.Bool

// NativeHistogramOptions holds the bucket configuration options for native histograms.
type NativeHistogramOptions struct {
	BucketFactor    float64
	MaxBucketNumber uint32
}

// defaultNativeHistogramOptions are the defaults applied to all histogram metrics
// when native histograms are enabled.
var defaultNativeHistogramOptions = NativeHistogramOptions{
	// BucketFactor is the growth factor for native histogram buckets.
	// A value of 1.1 means each bucket is at most 10% wider than the previous one.
	BucketFactor: 1.1,
	// MaxBucketNumber is the maximum number of buckets per native histogram.
	// Based on the OTel SDK recommendation for base2 exponential bucket histogram aggregation.
	MaxBucketNumber: 160,
}

// SetNativeHistogramsEnabled records whether native histograms are enabled.
// This should be called exactly once during component initialisation, driven
// by the NativeHistograms feature gate.
func SetNativeHistogramsEnabled(enabled bool) {
	nativeHistogramsEnabled.Store(enabled)
}

// NativeHistogramsEnabled reports whether native histograms are currently enabled.
func NativeHistogramsEnabled() bool {
	return nativeHistogramsEnabled.Load()
}

// NativeHistogramConfig returns the default bucket configuration for native histograms.
func NativeHistogramConfig() NativeHistogramOptions {
	return defaultNativeHistogramOptions
}
