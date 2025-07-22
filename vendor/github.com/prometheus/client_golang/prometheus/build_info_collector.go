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

package prometheus

import "runtime/debug"

// NewBuildInfoCollector is the obsolete version of collectors.NewBuildInfoCollector.
// See there for documentation.
//
// Deprecated: Use collectors.NewBuildInfoCollector instead.
func NewBuildInfoCollector() Collector {
	path, version, sum := "unknown", "unknown", "unknown"
	if bi, ok := debug.ReadBuildInfo(); ok {
		path = bi.Main.Path
		version = bi.Main.Version
		sum = bi.Main.Sum
	}
	c := &selfCollector{MustNewConstMetric(
		NewDesc(
			"go_build_info",
			"Build information about the main Go module.",
			nil, Labels{"path": path, "version": version, "checksum": sum},
		),
		GaugeValue, 1)}
	c.init(c.self)
	return c
}
