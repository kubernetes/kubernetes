// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"time"
)

// Trace allows internal access to some trace functionality.
// TODO(#412): remove this
var Trace interface{}

// LocalSpanStoreEnabled true if the local span store is enabled.
var LocalSpanStoreEnabled bool

// BucketConfiguration stores the number of samples to store for span buckets
// for successful and failed spans for a particular span name.
type BucketConfiguration struct {
	Name                 string
	MaxRequestsSucceeded int
	MaxRequestsErrors    int
}

// PerMethodSummary is a summary of the spans stored for a single span name.
type PerMethodSummary struct {
	Active         int
	LatencyBuckets []LatencyBucketSummary
	ErrorBuckets   []ErrorBucketSummary
}

// LatencyBucketSummary is a summary of a latency bucket.
type LatencyBucketSummary struct {
	MinLatency, MaxLatency time.Duration
	Size                   int
}

// ErrorBucketSummary is a summary of an error bucket.
type ErrorBucketSummary struct {
	ErrorCode int32
	Size      int
}
