// +build !libpfm !cgo

// Copyright 2020 Google Inc. All Rights Reserved.
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

// Collector of perf events for a container.
package perf

import (
	"github.com/google/cadvisor/stats"

	"k8s.io/klog/v2"
)

func NewCollector(cgroupPath string, events Events, numCores int) stats.Collector {
	return &stats.NoopCollector{}
}

// Finalize terminates libpfm4 to free resources.
func Finalize() {
	klog.V(1).Info("cAdvisor is build without cgo and/or libpfm support. Nothing to be finalized")
}
