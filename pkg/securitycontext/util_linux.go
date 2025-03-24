/*
Copyright 2025 The Kubernetes Authors.

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

package securitycontext

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// possibleCPUs returns the number of possible CPUs on this host.
func possibleCPUs() (cpus []int) {
	if ncpu := possibleCPUsParsed(); ncpu != nil {
		return ncpu
	}

	for i := range runtime.NumCPU() {
		cpus = append(cpus, i)
	}

	return cpus
}

// possibleCPUsParsed is parsing the amount of possible CPUs on this host from
// /sys/devices.
var possibleCPUsParsed = sync.OnceValue(func() (cpus []int) {
	data, err := os.ReadFile("/sys/devices/system/cpu/possible")
	if err != nil {
		return nil
	}

	ranges := strings.Split(strings.TrimSpace(string(data)), ",")

	for _, r := range ranges {
		if rStart, rEnd, ok := strings.Cut(r, "-"); !ok {
			cpu, err := strconv.Atoi(rStart)
			if err != nil {
				return nil
			}
			cpus = append(cpus, cpu)
		} else {
			var start, end int
			start, err := strconv.Atoi(rStart)
			if err != nil {
				return nil
			}
			end, err = strconv.Atoi(rEnd)
			if err != nil {
				return nil
			}
			for i := start; i <= end; i++ {
				cpus = append(cpus, i)
			}
		}
	}

	return cpus
})
