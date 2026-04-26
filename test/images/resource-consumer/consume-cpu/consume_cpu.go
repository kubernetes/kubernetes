//go:build !windows

/*
Copyright 2015 The Kubernetes Authors.

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

package main

import (
	"flag"
	"syscall"
	"time"
)

type procCPUSample struct {
	total float64
	t     time.Time
}

func getProcCPUSample() procCPUSample {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	user := float64(ru.Utime.Usec)/1e6 + float64(ru.Utime.Sec)
	sys := float64(ru.Stime.Usec)/1e6 + float64(ru.Stime.Sec)
	return procCPUSample{total: user + sys, t: time.Now()}
}

func getProcCPUTotalPct(first, second procCPUSample) float64 {
	dT := second.t.Sub(first.t).Seconds()
	if dT <= 0 {
		return 0
	}
	return 100 * (second.total - first.total) / dT
}

func main() {
	flag.Parse()
	// convert millicores to percentage
	millicoresPct := float64(*millicores) / float64(10)
	duration := time.Duration(*durationSec) * time.Second
	start := time.Now()
	first := getProcCPUSample()
	for time.Since(start) < duration {
		if getProcCPUTotalPct(first, getProcCPUSample()) < millicoresPct {
			doSomething()
		} else {
			time.Sleep(sleep)
		}
	}
}
