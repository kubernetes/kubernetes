//go:build windows

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

package resourceconsumer

import (
	"log"
	"syscall"
	"time"

	syswin "golang.org/x/sys/windows"
)

type procCPUStats struct {
	User   int64
	System int64
	Time   time.Time
	Total  int64
}

func statsNow(handle syscall.Handle) (s procCPUStats) {
	var processInfo syscall.Rusage
	syscall.GetProcessTimes(handle, &processInfo.CreationTime, &processInfo.ExitTime, &processInfo.KernelTime, &processInfo.UserTime)
	s.Time = time.Now()
	s.User = processInfo.UserTime.Nanoseconds()
	s.System = processInfo.KernelTime.Nanoseconds()
	s.Total = s.User + s.System
	return s
}

func usageNow(first procCPUStats, second procCPUStats) int64 {
	dT := second.Time.Sub(first.Time).Nanoseconds()
	dUsage := second.Total - first.Total
	if dT == 0 {
		return 0
	}
	return 1000 * dUsage / dT
}

// ConsumeCPU consumes a given number of millicores for the specified duration.
func ConsumeCPU(millicores int, durationSec int) {
	log.Printf("ConsumeCPU millicores: %v, durationSec: %v", millicores, durationSec)
	phandle, err := syswin.GetCurrentProcess()
	if err != nil {
		panic(err)
	}
	handle := syscall.Handle(phandle)

	duration := time.Duration(durationSec) * time.Second
	start := time.Now()
	first := statsNow(handle)
	for time.Since(start) < duration {
		currentMillicores := usageNow(first, statsNow(handle))
		if currentMillicores < int64(millicores) {
			doSomething()
		} else {
			time.Sleep(sleep)
		}
	}
}
