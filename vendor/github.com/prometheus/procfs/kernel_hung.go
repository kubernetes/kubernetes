// Copyright The Prometheus Authors
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

//go:build !windows

package procfs

import (
	"os"
	"strconv"
	"strings"
)

// KernelHung contains information about to the kernel's hung_task_detect_count number.
type KernelHung struct {
	// Indicates the total number of tasks that have been detected as hung since the system boot.
	// This file shows up if `CONFIG_DETECT_HUNG_TASK` is enabled.
	HungTaskDetectCount *uint64
}

// KernelHung returns values from /proc/sys/kernel/hung_task_detect_count.
func (fs FS) KernelHung() (KernelHung, error) {
	data, err := os.ReadFile(fs.proc.Path("sys", "kernel", "hung_task_detect_count"))
	if err != nil {
		return KernelHung{}, err
	}
	val, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return KernelHung{}, err
	}
	return KernelHung{
		HungTaskDetectCount: &val,
	}, nil
}
