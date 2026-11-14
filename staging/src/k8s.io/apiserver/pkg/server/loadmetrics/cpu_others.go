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

// cpuTimeSupported reports whether process CPU time measurement is supported on this platform.
// Returning false causes NewLoadReporter to fail apiserver startup when
// --enable-endpoint-load-metrics is set, rather than silently reporting 0% CPU utilization.
func cpuTimeSupported() bool {
	return false
}

// getProcessCPUTimeSeconds returns an error on non-Linux platforms where
// CLOCK_PROCESS_CPUTIME_ID is not implemented.
func getProcessCPUTimeSeconds() (float64, error) {
	return 0, fmt.Errorf("process CPU time measurement is not supported on %s", runtime.GOOS)
}
