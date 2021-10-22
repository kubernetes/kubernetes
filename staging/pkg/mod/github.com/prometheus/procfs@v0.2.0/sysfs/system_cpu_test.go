// Copyright 2018 The Prometheus Authors
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

// +build !windows

package sysfs

import (
	"reflect"
	"testing"
)

func makeUint64(v uint64) *uint64 {
	return &v
}

func TestCPUTopology(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}
	cpus, err := fs.CPUs()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := 2, len(cpus); want != have {
		t.Errorf("incorrect number of CPUs, have %v, want %v", want, have)
	}
	if want, have := "0", cpus[0].Number(); want != have {
		t.Errorf("incorrect name, have %v, want %v", want, have)
	}
	cpu0Topology, err := cpus[0].Topology()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "0", cpu0Topology.CoreID; want != have {
		t.Errorf("incorrect core ID, have %v, want %v", want, have)
	}
	if want, have := "0-7", cpu0Topology.CoreSiblingsList; want != have {
		t.Errorf("incorrect core siblings list, have %v, want %v", want, have)
	}
	cpu1Topology, err := cpus[1].Topology()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "0", cpu1Topology.PhysicalPackageID; want != have {
		t.Errorf("incorrect package ID, have %v, want %v", want, have)
	}
	if want, have := "1,5", cpu1Topology.ThreadSiblingsList; want != have {
		t.Errorf("incorrect thread siblings list, have %v, want %v", want, have)
	}
}

func TestCPUThermalThrottle(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}
	cpus, err := fs.CPUs()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := 2, len(cpus); want != have {
		t.Errorf("incorrect number of CPUs, have %v, want %v", want, have)
	}
	cpu0Throttle, err := cpus[0].ThermalThrottle()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := uint64(34818), cpu0Throttle.PackageThrottleCount; want != have {
		t.Errorf("incorrect package throttle count, have %v, want %v", want, have)
	}

	cpu1Throttle, err := cpus[1].ThermalThrottle()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := uint64(523), cpu1Throttle.CoreThrottleCount; want != have {
		t.Errorf("incorrect core throttle count, have %v, want %v", want, have)
	}
}

func TestSystemCpufreq(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	c, err := fs.SystemCpufreq()
	if err != nil {
		t.Fatal(err)
	}

	systemCpufreq := []SystemCPUCpufreqStats{
		// Has missing `cpuinfo_cur_freq` file.
		{
			Name:                     "0",
			CpuinfoCurrentFrequency:  nil,
			CpuinfoMinimumFrequency:  makeUint64(800000),
			CpuinfoMaximumFrequency:  makeUint64(2400000),
			CpuinfoTransitionLatency: makeUint64(0),
			ScalingCurrentFrequency:  makeUint64(1219917),
			ScalingMinimumFrequency:  makeUint64(800000),
			ScalingMaximumFrequency:  makeUint64(2400000),
			AvailableGovernors:       "performance powersave",
			Driver:                   "intel_pstate",
			Governor:                 "powersave",
			RelatedCpus:              "0",
			SetSpeed:                 "<unsupported>",
		},
		// Has missing `scaling_cur_freq` file.
		{
			Name:                     "1",
			CpuinfoCurrentFrequency:  makeUint64(1200195),
			CpuinfoMinimumFrequency:  makeUint64(1200000),
			CpuinfoMaximumFrequency:  makeUint64(3300000),
			CpuinfoTransitionLatency: makeUint64(4294967295),
			ScalingCurrentFrequency:  nil,
			ScalingMinimumFrequency:  makeUint64(1200000),
			ScalingMaximumFrequency:  makeUint64(3300000),
			AvailableGovernors:       "performance powersave",
			Driver:                   "intel_pstate",
			Governor:                 "powersave",
			RelatedCpus:              "1",
			SetSpeed:                 "<unsupported>",
		},
	}

	if !reflect.DeepEqual(systemCpufreq, c) {
		t.Errorf("Result not correct: want %v, have %v", systemCpufreq, c)
	}
}
