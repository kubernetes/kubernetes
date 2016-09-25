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
	"fmt"
	"math/rand"
	"time"

	info "github.com/google/cadvisor/info/v1"
)

func GenerateRandomStats(numStats, numCores int, duration time.Duration) []*info.ContainerStats {
	ret := make([]*info.ContainerStats, numStats)
	perCoreUsages := make([]uint64, numCores)
	currentTime := time.Now()
	for i := range perCoreUsages {
		perCoreUsages[i] = uint64(rand.Int63n(1000))
	}
	for i := 0; i < numStats; i++ {
		stats := new(info.ContainerStats)
		stats.Timestamp = currentTime
		currentTime = currentTime.Add(duration)

		percore := make([]uint64, numCores)
		for i := range perCoreUsages {
			perCoreUsages[i] += uint64(rand.Int63n(1000))
			percore[i] = perCoreUsages[i]
			stats.Cpu.Usage.Total += percore[i]
		}
		stats.Cpu.Usage.PerCpu = percore
		stats.Cpu.Usage.User = stats.Cpu.Usage.Total
		stats.Cpu.Usage.System = 0
		stats.Memory.Usage = uint64(rand.Int63n(4096))
		stats.Memory.Cache = uint64(rand.Int63n(4096))
		stats.Memory.RSS = uint64(rand.Int63n(4096))
		ret[i] = stats
	}
	return ret
}

func GenerateRandomContainerSpec(numCores int) info.ContainerSpec {
	ret := info.ContainerSpec{
		CreationTime: time.Now(),
		HasCpu:       true,
		Cpu:          info.CpuSpec{},
		HasMemory:    true,
		Memory:       info.MemorySpec{},
	}
	ret.Cpu.Limit = uint64(1000 + rand.Int63n(2000))
	ret.Cpu.MaxLimit = uint64(1000 + rand.Int63n(2000))
	ret.Cpu.Mask = fmt.Sprintf("0-%d", numCores-1)
	ret.Memory.Limit = uint64(4096 + rand.Int63n(4096))
	return ret
}

func GenerateRandomContainerInfo(containerName string, numCores int, query *info.ContainerInfoRequest, duration time.Duration) *info.ContainerInfo {
	stats := GenerateRandomStats(query.NumStats, numCores, duration)
	spec := GenerateRandomContainerSpec(numCores)

	ret := &info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name: containerName,
		},
		Spec:  spec,
		Stats: stats,
	}
	return ret
}
