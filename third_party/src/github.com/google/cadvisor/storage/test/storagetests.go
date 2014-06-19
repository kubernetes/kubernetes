// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/storage"
)

func buildTrace(cpu, mem []uint64, duration time.Duration) []*info.ContainerStats {
	if len(cpu) != len(mem) {
		panic("len(cpu) != len(mem)")
	}

	ret := make([]*info.ContainerStats, len(cpu))
	currentTime := time.Now()

	var cpuTotalUsage uint64 = 0
	for i, cpuUsage := range cpu {
		cpuTotalUsage += cpuUsage
		stats := new(info.ContainerStats)
		stats.Cpu = new(info.CpuStats)
		stats.Memory = new(info.MemoryStats)
		stats.Timestamp = currentTime
		currentTime = currentTime.Add(duration)

		stats.Cpu.Usage.Total = cpuTotalUsage
		stats.Cpu.Usage.User = stats.Cpu.Usage.Total
		stats.Cpu.Usage.System = 0

		stats.Memory.Usage = mem[i]

		ret[i] = stats
	}
	return ret
}

// The underlying driver must be able to hold more than 10 samples.
func StorageDriverTestSampleCpuUsage(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	N := 10
	cpuTrace := make([]uint64, 0, N)
	memTrace := make([]uint64, 0, N)

	// We need N+1 observations to get N samples
	for i := 0; i < N+1; i++ {
		cpuTrace = append(cpuTrace, uint64(rand.Intn(1000)))
		memTrace = append(memTrace, uint64(rand.Intn(1000)))
	}

	samplePeriod := 1 * time.Second

	ref := info.ContainerReference{
		Name: "container",
	}

	trace := buildTrace(cpuTrace, memTrace, samplePeriod)

	for _, stats := range trace {
		driver.AddStats(ref, stats)
	}

	samples, err := driver.Samples(ref.Name, N)
	if err != nil {
		t.Errorf("unable to sample stats: %v", err)
	}
	for _, sample := range samples {
		if sample.Duration != samplePeriod {
			t.Errorf("sample duration is %v, not %v", sample.Duration, samplePeriod)
		}
		cpuUsage := sample.Cpu.Usage
		found := false
		for _, u := range cpuTrace {
			if u == cpuUsage {
				found = true
			}
		}
		if !found {
			t.Errorf("unable to find cpu usage %v", cpuUsage)
		}
	}
}

func StorageDriverTestMaxMemoryUsage(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	N := 100
	memTrace := make([]uint64, N)
	cpuTrace := make([]uint64, N)
	for i := 0; i < N; i++ {
		memTrace[i] = uint64(i + 1)
		cpuTrace[i] = uint64(1)
	}

	ref := info.ContainerReference{
		Name: "container",
	}

	trace := buildTrace(cpuTrace, memTrace, 1*time.Second)

	for _, stats := range trace {
		driver.AddStats(ref, stats)
	}

	percentiles, err := driver.Percentiles(ref.Name, []int{50}, []int{50})
	if err != nil {
		t.Errorf("unable to call Percentiles(): %v", err)
	}
	maxUsage := uint64(N)
	if percentiles.MaxMemoryUsage != maxUsage {
		t.Fatalf("Max memory usage should be %v; received %v", maxUsage, percentiles.MaxMemoryUsage)
	}
}

func StorageDriverTestSamplesWithoutSample(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	trace := buildTrace(
		[]uint64{10},
		[]uint64{10},
		1*time.Second)
	ref := info.ContainerReference{
		Name: "container",
	}
	driver.AddStats(ref, trace[0])
	samples, err := driver.Samples(ref.Name, -1)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 0 {
		t.Errorf("There should be no sample")
	}
}

func StorageDriverTestPercentilesWithoutSample(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	trace := buildTrace(
		[]uint64{10},
		[]uint64{10},
		1*time.Second)
	ref := info.ContainerReference{
		Name: "container",
	}
	driver.AddStats(ref, trace[0])
	percentiles, err := driver.Percentiles(
		ref.Name,
		[]int{50},
		[]int{50},
	)
	if err != nil {
		t.Fatal(err)
	}
	if percentiles != nil {
		t.Errorf("There should be no percentiles")
	}
}
