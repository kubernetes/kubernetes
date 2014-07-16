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
	"reflect"
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
		stats.Cpu.Usage.PerCpu = []uint64{cpuTotalUsage}

		stats.Memory.Usage = mem[i]

		ret[i] = stats
	}
	return ret
}

func timeEq(t1, t2 time.Time, tolerance time.Duration) bool {
	// t1 should not be later than t2
	if t1.After(t2) {
		t1, t2 = t2, t1
	}
	diff := t2.Sub(t1)
	if diff <= tolerance {
		return true
	}
	return false
}

func durationEq(a, b time.Duration, tolerance time.Duration) bool {
	if a > b {
		a, b = b, a
	}
	diff := a - b
	if diff <= tolerance {
		return true
	}
	return false
}

const (
	// 10ms, i.e. 0.01s
	timePrecision time.Duration = 10 * time.Millisecond
)

// This function is useful because we do not require precise time
// representation.
func statsEq(a, b *info.ContainerStats) bool {
	if !timeEq(a.Timestamp, b.Timestamp, timePrecision) {
		return false
	}
	if !reflect.DeepEqual(a.Cpu, b.Cpu) {
		return false
	}
	if !reflect.DeepEqual(a.Memory, b.Memory) {
		return false
	}
	return true
}

// This function is useful because we do not require precise time
// representation.
func sampleEq(a, b *info.ContainerStatsSample) bool {
	if !timeEq(a.Timestamp, b.Timestamp, timePrecision) {
		return false
	}
	if !durationEq(a.Duration, b.Duration, timePrecision) {
		return false
	}
	if !reflect.DeepEqual(a.Cpu, b.Cpu) {
		return false
	}
	if !reflect.DeepEqual(a.Memory, b.Memory) {
		return false
	}
	return true
}

func samplesInTrace(samples []*info.ContainerStatsSample, cpuTrace, memTrace []uint64, samplePeriod time.Duration, t *testing.T) {
	for _, sample := range samples {
		if sample.Duration != samplePeriod {
			t.Errorf("sample duration is %v, not %v", sample.Duration, samplePeriod)
		}
		cpuUsage := sample.Cpu.Usage
		memUsage := sample.Memory.Usage
		found := false
		for _, u := range cpuTrace {
			if u == cpuUsage {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("unable to find cpu usage %v", cpuUsage)
		}
		found = false
		for _, u := range memTrace {
			if u == memUsage {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("unable to find mem usage %v", memUsage)
		}
	}
}

// This function will generate random stats and write
// them into the storage. The function will not close the driver
func StorageDriverFillRandomStatsFunc(
	containerName string,
	N int,
	driver storage.StorageDriver,
	t *testing.T,
) {
	cpuTrace := make([]uint64, 0, N)
	memTrace := make([]uint64, 0, N)

	// We need N+1 observations to get N samples
	for i := 0; i < N+1; i++ {
		cpuTrace = append(cpuTrace, uint64(rand.Intn(1000)))
		memTrace = append(memTrace, uint64(rand.Intn(1000)))
	}

	samplePeriod := 1 * time.Second

	ref := info.ContainerReference{
		Name: containerName,
	}

	trace := buildTrace(cpuTrace, memTrace, samplePeriod)

	for _, stats := range trace {
		err := driver.AddStats(ref, stats)
		if err != nil {
			t.Fatalf("unable to add stats: %v", err)
		}
	}
}

func StorageDriverTestSampleCpuUsage(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	N := 100
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
		err := driver.AddStats(ref, stats)
		if err != nil {
			t.Fatalf("unable to add stats: %v", err)
		}
		// set the trace to something else. The stats stored in the
		// storage should not be affected.
		stats.Cpu.Usage.Total = 0
		stats.Cpu.Usage.System = 0
		stats.Cpu.Usage.User = 0
	}

	samples, err := driver.Samples(ref.Name, N)
	if err != nil {
		t.Errorf("unable to sample stats: %v", err)
	}
	if len(samples) == 0 {
		t.Fatal("should at least store one sample")
	}
	samplesInTrace(samples, cpuTrace, memTrace, samplePeriod, t)

	samples, err = driver.Samples(ref.Name, -1)
	if err != nil {
		t.Errorf("unable to sample stats: %v", err)
	}
	samplesInTrace(samples, cpuTrace, memTrace, samplePeriod, t)

	samples, err = driver.Samples(ref.Name, N-5)
	if err != nil {
		t.Errorf("unable to sample stats: %v", err)
	}
	samplesInTrace(samples, cpuTrace, memTrace, samplePeriod, t)
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
		err := driver.AddStats(ref, stats)
		if err != nil {
			t.Fatalf("unable to add stats: %v", err)
		}
		// set the trace to something else. The stats stored in the
		// storage should not be affected.
		stats.Cpu.Usage.Total = 0
		stats.Cpu.Usage.System = 0
		stats.Cpu.Usage.User = 0
		stats.Memory.Usage = 0
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

func StorageDriverTestPercentiles(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	N := 100
	cpuTrace := make([]uint64, N)
	memTrace := make([]uint64, N)
	for i := 1; i < N+1; i++ {
		cpuTrace[i-1] = uint64(i)
		memTrace[i-1] = uint64(i)
	}

	trace := buildTrace(cpuTrace, memTrace, 1*time.Second)

	ref := info.ContainerReference{
		Name: "container",
	}
	for _, stats := range trace {
		driver.AddStats(ref, stats)
	}
	percentages := []int{
		80,
		90,
		50,
	}
	percentiles, err := driver.Percentiles(ref.Name, percentages, percentages)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range percentiles.CpuUsagePercentiles {
		for _, y := range percentiles.CpuUsagePercentiles {
			// lower percentage, smaller value
			if x.Percentage < y.Percentage && x.Value > y.Value {
				t.Errorf("%v percent is %v; while %v percent is %v",
					x.Percentage, x.Value, y.Percentage, y.Value)
			}
		}
	}
	for _, x := range percentiles.MemoryUsagePercentiles {
		for _, y := range percentiles.MemoryUsagePercentiles {
			if x.Percentage < y.Percentage && x.Value > y.Value {
				t.Errorf("%v percent is %v; while %v percent is %v",
					x.Percentage, x.Value, y.Percentage, y.Value)
			}
		}
	}
}

func StorageDriverTestRetrievePartialRecentStats(driver storage.StorageDriver, t *testing.T) {
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

	recentStats, err := driver.RecentStats(ref.Name, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(recentStats) == 0 {
		t.Fatal("should at least store one stats")
	}

	if len(recentStats) > 10 {
		t.Fatalf("returned %v stats, not 10.", len(recentStats))
	}

	actualRecentStats := trace[len(trace)-len(recentStats):]

	// The returned stats should be sorted in time increasing order
	for i, s := range actualRecentStats {
		r := recentStats[i]
		if !statsEq(s, r) {
			t.Errorf("unexpected stats %+v with memory usage %v", r, r.Memory.Usage)
		}
	}
}

func StorageDriverTestRetrieveAllRecentStats(driver storage.StorageDriver, t *testing.T) {
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

	recentStats, err := driver.RecentStats(ref.Name, -1)
	if err != nil {
		t.Fatal(err)
	}
	if len(recentStats) == 0 {
		t.Fatal("should at least store one stats")
	}
	if len(recentStats) > N {
		t.Fatalf("returned %v stats, not 100.", len(recentStats))
	}

	actualRecentStats := trace[len(trace)-len(recentStats):]

	// The returned stats should be sorted in time increasing order
	for i, s := range actualRecentStats {
		r := recentStats[i]
		if !statsEq(s, r) {
			t.Errorf("unexpected stats %+v with memory usage %v", r, r.Memory.Usage)
		}
	}
}

func StorageDriverTestNoRecentStats(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	nonExistContainer := "somerandomecontainer"
	stats, _ := driver.RecentStats(nonExistContainer, -1)
	if len(stats) > 0 {
		t.Errorf("RecentStats() returns %v stats on non exist container", len(stats))
	}
}

func StorageDriverTestNoSamples(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	nonExistContainer := "somerandomecontainer"
	samples, _ := driver.Samples(nonExistContainer, -1)
	if len(samples) > 0 {
		t.Errorf("Samples() returns %v samples on non exist container", len(samples))
	}
}

func StorageDriverTestPercentilesWithoutStats(driver storage.StorageDriver, t *testing.T) {
	defer driver.Close()
	nonExistContainer := "somerandomecontainer"
	percentiles, _ := driver.Percentiles(nonExistContainer, []int{50, 80}, []int{50, 80})
	if percentiles == nil {
		return
	}
	if percentiles.MaxMemoryUsage != 0 {
		t.Errorf("Percentiles() reports max memory usage > 0 when there's no stats.")
	}
	for _, p := range percentiles.CpuUsagePercentiles {
		if p.Value != 0 {
			t.Errorf("Percentiles() reports cpu usage is %v when there's no stats.", p.Value)
		}
	}
	for _, p := range percentiles.MemoryUsagePercentiles {
		if p.Value != 0 {
			t.Errorf("Percentiles() reports memory usage is %v when there's no stats.", p.Value)
		}
	}
}
