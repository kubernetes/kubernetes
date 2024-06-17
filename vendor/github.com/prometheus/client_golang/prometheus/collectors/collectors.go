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

// Package collectors provides implementations of prometheus.Collector to
// conveniently collect process and Go-related metrics.
package collectors

import "github.com/prometheus/client_golang/prometheus"

// NewBuildInfoCollector returns a collector collecting a single metric
// "go_build_info" with the constant value 1 and three labels "path", "version",
// and "checksum". Their label values contain the main module path, version, and
// checksum, respectively. The labels will only have meaningful values if the
// binary is built with Go module support and from source code retrieved from
// the source repository (rather than the local file system). This is usually
// accomplished by building from outside of GOPATH, specifying the full address
// of the main package, e.g. "GO111MODULE=on go run
// github.com/prometheus/client_golang/examples/random". If built without Go
// module support, all label values will be "unknown". If built with Go module
// support but using the source code from the local file system, the "path" will
// be set appropriately, but "checksum" will be empty and "version" will be
// "(devel)".
//
// This collector uses only the build information for the main module. See
// https://github.com/povilasv/prommod for an example of a collector for the
// module dependencies.
func NewBuildInfoCollector() prometheus.Collector {
	//nolint:staticcheck // Ignore SA1019 until v2.
	return prometheus.NewBuildInfoCollector()
}
