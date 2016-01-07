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

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
)

type TestStorageDriver interface {
	StatsEq(a *info.ContainerStats, b *info.ContainerStats) bool
	storage.StorageDriver
}

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
		stats.Timestamp = currentTime
		currentTime = currentTime.Add(duration)

		stats.Cpu.Usage.Total = cpuTotalUsage
		stats.Cpu.Usage.User = stats.Cpu.Usage.Total
		stats.Cpu.Usage.System = 0
		stats.Cpu.Usage.PerCpu = []uint64{cpuTotalUsage}

		stats.Memory.Usage = mem[i]

		stats.Network.RxBytes = uint64(rand.Intn(10000))
		stats.Network.RxErrors = uint64(rand.Intn(1000))
		stats.Network.TxBytes = uint64(rand.Intn(100000))
		stats.Network.TxErrors = uint64(rand.Intn(1000))

		stats.Filesystem = make([]info.FsStats, 1)
		stats.Filesystem[0].Device = "/dev/sda1"
		stats.Filesystem[0].Limit = 1024000000
		stats.Filesystem[0].Usage = 1024000
		ret[i] = stats
	}
	return ret
}

func TimeEq(t1, t2 time.Time, tolerance time.Duration) bool {
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

const (
	// 10ms, i.e. 0.01s
	timePrecision time.Duration = 10 * time.Millisecond
)

// This function is useful because we do not require precise time
// representation.
func DefaultStatsEq(a, b *info.ContainerStats) bool {
	if !TimeEq(a.Timestamp, b.Timestamp, timePrecision) {
		return false
	}
	if !reflect.DeepEqual(a.Cpu, b.Cpu) {
		return false
	}
	if !reflect.DeepEqual(a.Memory, b.Memory) {
		return false
	}
	if !reflect.DeepEqual(a.Network, b.Network) {
		return false
	}
	if !reflect.DeepEqual(a.Filesystem, b.Filesystem) {
		return false
	}

	return true
}

// This function will generate random stats and write
// them into the storage. The function will not close the driver
func StorageDriverFillRandomStatsFunc(
	containerName string,
	N int,
	driver TestStorageDriver,
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
