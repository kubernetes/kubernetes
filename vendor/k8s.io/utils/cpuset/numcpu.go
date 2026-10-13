/*
Copyright 2026 The Kubernetes Authors.

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

package cpuset

import (
	"fmt"
	"os"
	"strings"
)

// onlineCPUsPath is the sysfs file listing the host's online CPUs. It is a
// variable so tests can point it at a fixture.
var onlineCPUsPath = "/sys/devices/system/cpu/online"

// NumCPU returns the number of CPUs that are online on the host, read from
// /sys/devices/system/cpu/online. The file holds a cpu list such as "0-3,8-11";
// see https://docs.kernel.org/admin-guide/cputopology.html for the format.
//
// Unlike runtime.NumCPU, the result is not affected by the calling process's
// CPU affinity (its cgroup cpuset), so it reflects the whole machine. This is
// what you want when sizing a host-wide resource by CPU count.
//
// NumCPU returns an error if the online file cannot be read or parsed, for
// example on non-Linux platforms. Callers that need a value regardless should
// fall back to runtime.NumCPU.
func NumCPU() (int, error) {
	data, err := os.ReadFile(onlineCPUsPath)
	if err != nil {
		return 0, err
	}
	set, err := Parse(strings.TrimSpace(string(data)))
	if err != nil {
		return 0, fmt.Errorf("parsing %q: %w", onlineCPUsPath, err)
	}
	return set.Size(), nil
}
