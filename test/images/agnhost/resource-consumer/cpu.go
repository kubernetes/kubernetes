//go:build !windows

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
	"time"

	"bitbucket.org/bertimus9/systemstat"
)

// ConsumeCPU consumes a given number of millicores for the specified duration.
func ConsumeCPU(millicores int, durationSec int) {
	log.Printf("ConsumeCPU millicores: %v, durationSec: %v", millicores, durationSec)
	millicoresPct := float64(millicores) / 10
	duration := time.Duration(durationSec) * time.Second
	start := time.Now()
	first := systemstat.GetProcCPUSample()
	for time.Since(start) < duration {
		cpu := systemstat.GetProcCPUAverage(first, systemstat.GetProcCPUSample(), systemstat.GetUptime().Uptime)
		if cpu.TotalPct < millicoresPct {
			doSomething()
		} else {
			time.Sleep(sleep)
		}
	}
}
