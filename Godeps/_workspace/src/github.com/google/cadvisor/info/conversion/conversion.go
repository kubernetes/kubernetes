// Copyright 2015 Google Inc. All Rights Reserved.
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

// Utilities for converting v1 structs to v2 structs.
package conversion

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
)

// Get V2 container spec from v1 container info.
func ConvertSpec(specV1 *v1.ContainerSpec, aliases []string, namespace string) v2.ContainerSpec {
	specV2 := v2.ContainerSpec{
		CreationTime:     specV1.CreationTime,
		HasCpu:           specV1.HasCpu,
		HasMemory:        specV1.HasMemory,
		HasFilesystem:    specV1.HasFilesystem,
		HasNetwork:       specV1.HasNetwork,
		HasDiskIo:        specV1.HasDiskIo,
		HasCustomMetrics: specV1.HasCustomMetrics,
		Image:            specV1.Image,
	}
	if specV1.HasCpu {
		specV2.Cpu.Limit = specV1.Cpu.Limit
		specV2.Cpu.MaxLimit = specV1.Cpu.MaxLimit
		specV2.Cpu.Mask = specV1.Cpu.Mask
	}
	if specV1.HasMemory {
		specV2.Memory.Limit = specV1.Memory.Limit
		specV2.Memory.Reservation = specV1.Memory.Reservation
		specV2.Memory.SwapLimit = specV1.Memory.SwapLimit
	}
	if specV1.HasCustomMetrics {
		specV2.CustomMetrics = specV1.CustomMetrics
	}
	specV2.Aliases = aliases
	specV2.Namespace = namespace
	return specV2
}

func ConvertStats(statsV1 []*v1.ContainerStats, specV1 *v1.ContainerSpec) []*v2.ContainerStats {
	stats := make([]*v2.ContainerStats, 0, len(statsV1))
	var last *v1.ContainerStats
	for _, val := range statsV1 {
		stat := v2.ContainerStats{
			Timestamp:        val.Timestamp,
			HasCpu:           specV1.HasCpu,
			HasMemory:        specV1.HasMemory,
			HasNetwork:       specV1.HasNetwork,
			HasFilesystem:    specV1.HasFilesystem,
			HasDiskIo:        specV1.HasDiskIo,
			HasCustomMetrics: specV1.HasCustomMetrics,
		}
		if stat.HasCpu {
			stat.Cpu = val.Cpu
			cpuInst, err := instCpuStats(last, val)
			if err != nil {
				glog.Warningf("Could not get instant cpu stats: %v", err)
			} else {
				stat.CpuInst = cpuInst
			}
			last = val
		}
		if stat.HasMemory {
			stat.Memory = val.Memory
		}
		if stat.HasNetwork {
			stat.Network.Interfaces = val.Network.Interfaces
		}
		if stat.HasFilesystem {
			stat.Filesystem = val.Filesystem
		}
		if stat.HasDiskIo {
			stat.DiskIo = val.DiskIo
		}
		if stat.HasCustomMetrics {
			stat.CustomMetrics = val.CustomMetrics
		}
		// TODO(rjnagal): Handle load stats.
		stats = append(stats, &stat)
	}
	return stats
}

func instCpuStats(last, cur *v1.ContainerStats) (*v2.CpuInstStats, error) {
	if last == nil {
		return nil, nil
	}
	if !cur.Timestamp.After(last.Timestamp) {
		return nil, fmt.Errorf("container stats move backwards in time")
	}
	if len(last.Cpu.Usage.PerCpu) != len(cur.Cpu.Usage.PerCpu) {
		return nil, fmt.Errorf("different number of cpus")
	}
	timeDelta := cur.Timestamp.Sub(last.Timestamp)
	if timeDelta <= 100*time.Millisecond {
		return nil, fmt.Errorf("time delta unexpectedly small")
	}
	// Nanoseconds to gain precision and avoid having zero seconds if the
	// difference between the timestamps is just under a second
	timeDeltaNs := uint64(timeDelta.Nanoseconds())
	convertToRate := func(lastValue, curValue uint64) (uint64, error) {
		if curValue < lastValue {
			return 0, fmt.Errorf("cumulative stats decrease")
		}
		valueDelta := curValue - lastValue
		return (valueDelta * 1e9) / timeDeltaNs, nil
	}
	total, err := convertToRate(last.Cpu.Usage.Total, cur.Cpu.Usage.Total)
	if err != nil {
		return nil, err
	}
	percpu := make([]uint64, len(last.Cpu.Usage.PerCpu))
	for i := range percpu {
		var err error
		percpu[i], err = convertToRate(last.Cpu.Usage.PerCpu[i], cur.Cpu.Usage.PerCpu[i])
		if err != nil {
			return nil, err
		}
	}
	user, err := convertToRate(last.Cpu.Usage.User, cur.Cpu.Usage.User)
	if err != nil {
		return nil, err
	}
	system, err := convertToRate(last.Cpu.Usage.System, cur.Cpu.Usage.System)
	if err != nil {
		return nil, err
	}
	return &v2.CpuInstStats{
		Usage: v2.CpuInstUsage{
			Total:  total,
			PerCpu: percpu,
			User:   user,
			System: system,
		},
	}, nil
}
