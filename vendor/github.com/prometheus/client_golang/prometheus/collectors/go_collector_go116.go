// Copyright 2021 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !go1.17
// +build !go1.17

package collectors

import "github.com/prometheus/client_golang/prometheus"

// NewGoCollector returns a collector that exports metrics about the current Go
// process. This includes memory stats. To collect those, runtime.ReadMemStats
// is called. This requires to “stop the world”, which usually only happens for
// garbage collection (GC). Take the following implications into account when
// deciding whether to use the Go collector:
//
// 1. The performance impact of stopping the world is the more relevant the more
// frequently metrics are collected. However, with Go1.9 or later the
// stop-the-world time per metrics collection is very short (~25µs) so that the
// performance impact will only matter in rare cases. However, with older Go
// versions, the stop-the-world duration depends on the heap size and can be
// quite significant (~1.7 ms/GiB as per
// https://go-review.googlesource.com/c/go/+/34937).
//
// 2. During an ongoing GC, nothing else can stop the world. Therefore, if the
// metrics collection happens to coincide with GC, it will only complete after
// GC has finished. Usually, GC is fast enough to not cause problems. However,
// with a very large heap, GC might take multiple seconds, which is enough to
// cause scrape timeouts in common setups. To avoid this problem, the Go
// collector will use the memstats from a previous collection if
// runtime.ReadMemStats takes more than 1s. However, if there are no previously
// collected memstats, or their collection is more than 5m ago, the collection
// will block until runtime.ReadMemStats succeeds.
//
// NOTE: The problem is solved in Go 1.15, see
// https://github.com/golang/go/issues/19812 for the related Go issue.
func NewGoCollector() prometheus.Collector {
	return prometheus.NewGoCollector()
}
