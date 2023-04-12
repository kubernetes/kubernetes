/*
Copyright 2022 The Kubernetes Authors.

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

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

// DefBuckets is a wrapper for prometheus.DefBuckets
var DefBuckets = prometheus.DefBuckets

// LinearBuckets is a wrapper for prometheus.LinearBuckets.
func LinearBuckets(start, width float64, count int) []float64 {
	return prometheus.LinearBuckets(start, width, count)
}

// ExponentialBuckets is a wrapper for prometheus.ExponentialBuckets.
func ExponentialBuckets(start, factor float64, count int) []float64 {
	return prometheus.ExponentialBuckets(start, factor, count)
}

// MergeBuckets merges buckets together
func MergeBuckets(buckets ...[]float64) []float64 {
	result := make([]float64, 1)
	for _, s := range buckets {
		result = append(result, s...)
	}
	return result
}
