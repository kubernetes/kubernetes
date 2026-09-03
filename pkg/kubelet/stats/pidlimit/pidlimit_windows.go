//go:build windows

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
	"time"
	"unsafe"

	"golang.org/x/sys/windows"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

// Stats reports the current system-wide process count on Windows.
//
// Unlike Linux, Windows has no kernel pid knob equivalent to kernel.pid_max,
// so there is no real ceiling to report. MaxPID is deliberately left unset:
// fabricating one (for example mirroring the current process count) would make
// the pid.available eviction signal (MaxPID - NumOfRunningProcesses) always
// evaluate to 0 and permanently drive the node into NodePIDPressure, evicting
// pods. Leaving MaxPID nil keeps pid.available eviction unsupported on
// Windows, matching the behavior for other non-Linux platforms, while still
// surfacing the live process count for observability.
func Stats() (*statsapi.RlimitStats, error) {
	rlimit := &statsapi.RlimitStats{}

	count, err := currentProcessCount()
	if err != nil {
		return nil, err
	}
	rlimit.NumOfRunningProcesses = &count

	rlimit.Time = v1.NewTime(time.Now())

	return rlimit, nil
}

// currentProcessCount enumerates all processes on the system via
// NtQuerySystemInformation and returns how many are currently running.
func currentProcessCount() (int64, error) {
	// NtQuerySystemInformation reports STATUS_INFO_LENGTH_MISMATCH when the
	// buffer is too small, so grow it until the enumeration fits.
	bufSize := 1024 * 1024
	for {
		buf := make([]byte, bufSize)
		var retLen uint32
		err := windows.NtQuerySystemInformation(windows.SystemProcessInformation, unsafe.Pointer(&buf[0]), uint32(len(buf)), &retLen)
		if err != nil && err != windows.STATUS_INFO_LENGTH_MISMATCH {
			return 0, err
		}
		if err == nil {
			count := int64(0)
			proc := (*windows.SYSTEM_PROCESS_INFORMATION)(unsafe.Pointer(&buf[0]))
			for {
				count++
				if proc.NextEntryOffset == 0 {
					return count, nil
				}
				proc = (*windows.SYSTEM_PROCESS_INFORMATION)(unsafe.Pointer(uintptr(unsafe.Pointer(proc)) + uintptr(proc.NextEntryOffset)))
			}
		}
		bufSize *= 2
	}
}
