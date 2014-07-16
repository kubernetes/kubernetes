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

package info

import (
	"reflect"
	"testing"
	"time"
)

func TestStatsStartTime(t *testing.T) {
	N := 10
	stats := make([]*ContainerStats, 0, N)
	ct := time.Now()
	for i := 0; i < N; i++ {
		s := &ContainerStats{
			Timestamp: ct.Add(time.Duration(i) * time.Second),
		}
		stats = append(stats, s)
	}
	cinfo := &ContainerInfo{
		ContainerReference: ContainerReference{
			Name: "/some/container",
		},
		Stats: stats,
	}
	ref := ct.Add(time.Duration(N-1) * time.Second)
	end := cinfo.StatsEndTime()

	if !ref.Equal(end) {
		t.Errorf("end time is %v; should be %v", end, ref)
	}
}

func TestStatsEndTime(t *testing.T) {
	N := 10
	stats := make([]*ContainerStats, 0, N)
	ct := time.Now()
	for i := 0; i < N; i++ {
		s := &ContainerStats{
			Timestamp: ct.Add(time.Duration(i) * time.Second),
		}
		stats = append(stats, s)
	}
	cinfo := &ContainerInfo{
		ContainerReference: ContainerReference{
			Name: "/some/container",
		},
		Stats: stats,
	}
	ref := ct
	start := cinfo.StatsStartTime()

	if !ref.Equal(start) {
		t.Errorf("start time is %v; should be %v", start, ref)
	}
}

func TestPercentiles(t *testing.T) {
	N := 100
	data := make([]uint64, N)

	for i := 1; i < N+1; i++ {
		data[i-1] = uint64(i)
	}
	percentages := []int{
		80,
		90,
		50,
	}
	percentiles := uint64Slice(data).Percentiles(percentages...)
	for _, s := range percentiles {
		if s.Value != uint64(s.Percentage) {
			t.Errorf("%v percentile data should be %v, but got %v", s.Percentage, s.Percentage, s.Value)
		}
	}
}

func TestPercentilesSmallDataSet(t *testing.T) {
	var value uint64 = 11
	data := []uint64{value}

	percentages := []int{
		80,
		90,
		50,
	}
	percentiles := uint64Slice(data).Percentiles(percentages...)
	for _, s := range percentiles {
		if s.Value != value {
			t.Errorf("%v percentile data should be %v, but got %v", s.Percentage, value, s.Value)
		}
	}
}

func TestNewSampleNilStats(t *testing.T) {
	stats := &ContainerStats{
		Cpu:    &CpuStats{},
		Memory: &MemoryStats{},
	}
	stats.Cpu.Usage.PerCpu = []uint64{uint64(10)}
	stats.Cpu.Usage.Total = uint64(10)
	stats.Cpu.Usage.System = uint64(2)
	stats.Cpu.Usage.User = uint64(8)
	stats.Memory.Usage = uint64(200)

	sample, err := NewSample(nil, stats)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}

	sample, err = NewSample(stats, nil)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}
}

func createStats(cpuUsage, memUsage uint64, timestamp time.Time) *ContainerStats {
	stats := &ContainerStats{
		Cpu:    &CpuStats{},
		Memory: &MemoryStats{},
	}
	stats.Cpu.Usage.PerCpu = []uint64{cpuUsage}
	stats.Cpu.Usage.Total = cpuUsage
	stats.Cpu.Usage.System = 0
	stats.Cpu.Usage.User = cpuUsage
	stats.Memory.Usage = memUsage
	stats.Timestamp = timestamp
	return stats
}

func TestAddSample(t *testing.T) {
	cpuPrevUsage := uint64(10)
	cpuCurrentUsage := uint64(15)
	memCurrentUsage := uint64(200)
	prevTime := time.Now()

	prev := createStats(cpuPrevUsage, memCurrentUsage, prevTime)
	current := createStats(cpuCurrentUsage, memCurrentUsage, prevTime.Add(1*time.Second))

	sample, err := NewSample(prev, current)
	if err != nil {
		t.Errorf("should be able to generate a sample. but received error: %v", err)
	}
	if sample == nil {
		t.Fatalf("nil sample and nil error. unexpected result!")
	}

	if sample.Memory.Usage != memCurrentUsage {
		t.Errorf("wrong memory usage: %v. should be %v", sample.Memory.Usage, memCurrentUsage)
	}

	if sample.Cpu.Usage != cpuCurrentUsage-cpuPrevUsage {
		t.Errorf("wrong CPU usage: %v. should be %v", sample.Cpu.Usage, cpuCurrentUsage-cpuPrevUsage)
	}
}

func TestAddSampleIncompleteStats(t *testing.T) {
	cpuPrevUsage := uint64(10)
	cpuCurrentUsage := uint64(15)
	memCurrentUsage := uint64(200)
	prevTime := time.Now()

	prev := createStats(cpuPrevUsage, memCurrentUsage, prevTime)
	current := createStats(cpuCurrentUsage, memCurrentUsage, prevTime.Add(1*time.Second))
	stats := &ContainerStats{
		Cpu:    prev.Cpu,
		Memory: nil,
	}
	sample, err := NewSample(stats, current)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}
	sample, err = NewSample(prev, stats)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}

	stats = &ContainerStats{
		Cpu:    nil,
		Memory: prev.Memory,
	}
	sample, err = NewSample(stats, current)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}
	sample, err = NewSample(prev, stats)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}
}

func TestAddSampleWrongOrder(t *testing.T) {
	cpuPrevUsage := uint64(10)
	cpuCurrentUsage := uint64(15)
	memCurrentUsage := uint64(200)
	prevTime := time.Now()

	prev := createStats(cpuPrevUsage, memCurrentUsage, prevTime)
	current := createStats(cpuCurrentUsage, memCurrentUsage, prevTime.Add(1*time.Second))

	sample, err := NewSample(current, prev)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}
}

func TestAddSampleWrongCpuUsage(t *testing.T) {
	cpuPrevUsage := uint64(15)
	cpuCurrentUsage := uint64(10)
	memCurrentUsage := uint64(200)
	prevTime := time.Now()

	prev := createStats(cpuPrevUsage, memCurrentUsage, prevTime)
	current := createStats(cpuCurrentUsage, memCurrentUsage, prevTime.Add(1*time.Second))

	sample, err := NewSample(prev, current)
	if err == nil {
		t.Errorf("generated an unexpected sample: %+v", sample)
	}
}

func TestAddSampleHotPluggingCpu(t *testing.T) {
	cpuPrevUsage := uint64(10)
	cpuCurrentUsage := uint64(15)
	memCurrentUsage := uint64(200)
	prevTime := time.Now()

	prev := createStats(cpuPrevUsage, memCurrentUsage, prevTime)
	current := createStats(cpuCurrentUsage, memCurrentUsage, prevTime.Add(1*time.Second))
	current.Cpu.Usage.PerCpu = append(current.Cpu.Usage.PerCpu, 10)

	sample, err := NewSample(prev, current)
	if err != nil {
		t.Errorf("should be able to generate a sample. but received error: %v", err)
	}
	if len(sample.Cpu.PerCpuUsage) != 2 {
		t.Fatalf("Should have 2 cores.")
	}
	if sample.Cpu.PerCpuUsage[0] != cpuCurrentUsage-cpuPrevUsage {
		t.Errorf("First cpu usage is %v. should be %v", sample.Cpu.PerCpuUsage[0], cpuCurrentUsage-cpuPrevUsage)
	}
	if sample.Cpu.PerCpuUsage[1] != 10 {
		t.Errorf("Second cpu usage is %v. should be 10", sample.Cpu.PerCpuUsage[1])
	}
}

func TestAddSampleHotUnpluggingCpu(t *testing.T) {
	cpuPrevUsage := uint64(10)
	cpuCurrentUsage := uint64(15)
	memCurrentUsage := uint64(200)
	prevTime := time.Now()

	prev := createStats(cpuPrevUsage, memCurrentUsage, prevTime)
	current := createStats(cpuCurrentUsage, memCurrentUsage, prevTime.Add(1*time.Second))
	prev.Cpu.Usage.PerCpu = append(prev.Cpu.Usage.PerCpu, 10)

	sample, err := NewSample(prev, current)
	if err != nil {
		t.Errorf("should be able to generate a sample. but received error: %v", err)
	}
	if len(sample.Cpu.PerCpuUsage) != 1 {
		t.Fatalf("Should have 1 cores.")
	}
	if sample.Cpu.PerCpuUsage[0] != cpuCurrentUsage-cpuPrevUsage {
		t.Errorf("First cpu usage is %v. should be %v", sample.Cpu.PerCpuUsage[0], cpuCurrentUsage-cpuPrevUsage)
	}
}

func TestContainerStatsCopy(t *testing.T) {
	stats := createStats(100, 101, time.Now())
	shadowStats := stats.Copy(nil)
	if !reflect.DeepEqual(stats, shadowStats) {
		t.Errorf("Copy() returned different object")
	}
	stats.Cpu.Usage.PerCpu[0] = shadowStats.Cpu.Usage.PerCpu[0] + 1
	stats.Cpu.Load = shadowStats.Cpu.Load + 1
	stats.Memory.Usage = shadowStats.Memory.Usage + 1
	if reflect.DeepEqual(stats, shadowStats) {
		t.Errorf("Copy() did not deeply copy the object")
	}
	stats = shadowStats.Copy(stats)
	if !reflect.DeepEqual(stats, shadowStats) {
		t.Errorf("Copy() returned different object")
	}
}
