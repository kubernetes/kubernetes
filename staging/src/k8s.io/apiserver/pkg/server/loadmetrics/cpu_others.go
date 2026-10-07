//go:build !linux

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

package loadmetrics

import (
	"fmt"
	"runtime"
)

// Failing NewLoadReporter on non-Linux platforms prevents apiserver startup
// when --enable-endpoint-load-metrics is set, rather than silently reporting
// 0% CPU utilization to load balancers.
func cpuTimeSupported() bool {
	return false
}

func getProcessCPUTimeSeconds() (float64, error) {
	return 0, fmt.Errorf("process CPU time measurement is not supported on %s", runtime.GOOS)
}
