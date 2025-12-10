//go:build linux
// +build linux

/*
Copyright 2017 The Kubernetes Authors.

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

package pidlimit

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

// Stats provides basic information about max and current process count
func Stats() (*statsapi.RlimitStats, error) {
	rlimit := &statsapi.RlimitStats{}

	taskMax := int64(-1)
	// Calculate the minimum of kernel.pid_max and kernel.threads-max as they both specify the
	// system-wide limit on the number of tasks.
	for _, file := range []string{"/proc/sys/kernel/pid_max", "/proc/sys/kernel/threads-max"} {
		if content, err := os.ReadFile(file); err == nil {
			if limit, err := strconv.ParseInt(string(content[:len(content)-1]), 10, 64); err == nil {
				if taskMax == -1 || taskMax > limit {
					taskMax = limit
				}
			}
		}
	}
	// Both reads did not fail.
	if taskMax >= 0 {
		rlimit.MaxPID = &taskMax
	}

	// Prefer to read "/proc/loadavg" when possible because sysinfo(2)
	// returns truncated number when greater than 65538. See
	// https://github.com/kubernetes/kubernetes/issues/107107
	if procs, err := runningTaskCount(); err == nil {
		rlimit.NumOfRunningProcesses = &procs
	} else {
		var info syscall.Sysinfo_t
		syscall.Sysinfo(&info)
		procs := int64(info.Procs)
		rlimit.NumOfRunningProcesses = &procs
	}

	rlimit.Time = v1.NewTime(time.Now())

	return rlimit, nil
}

func runningTaskCount() (int64, error) {
	// Example: 1.36 3.49 4.53 2/3518 3715089
	bytes, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, err
	}
	fields := strings.Fields(string(bytes))
	if len(fields) < 5 {
		return 0, fmt.Errorf("not enough fields in /proc/loadavg")
	}
	subfields := strings.Split(fields[3], "/")
	if len(subfields) != 2 {
		return 0, fmt.Errorf("error parsing fourth field of /proc/loadavg")
	}
	return strconv.ParseInt(subfields[1], 10, 64)
}
