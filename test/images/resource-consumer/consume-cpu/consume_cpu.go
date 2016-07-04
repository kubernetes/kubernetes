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
	"math"
	"time"

	"bitbucket.org/bertimus9/systemstat"
)

const sleep = time.Duration(10) * time.Millisecond

func doSomething() {
	for i := 1; i < 10000000; i++ {
		x := float64(0)
		x += math.Sqrt(0)
	}
}

var (
	millicores  = flag.Int("millicores", 0, "millicores number")
	durationSec = flag.Int("duration-sec", 0, "duration time in seconds")
)

func main() {
	flag.Parse()
	// convert millicores to percentage
	millicoresPct := float64(*millicores) / float64(10)
	duration := time.Duration(*durationSec) * time.Second
	start := time.Now()
	first := systemstat.GetProcCPUSample()
	for time.Now().Sub(start) < duration {
		cpu := systemstat.GetProcCPUAverage(first, systemstat.GetProcCPUSample(), systemstat.GetUptime().Uptime)
		if cpu.TotalPct < millicoresPct {
			doSomething()
		} else {
			time.Sleep(sleep)
		}
	}
}
