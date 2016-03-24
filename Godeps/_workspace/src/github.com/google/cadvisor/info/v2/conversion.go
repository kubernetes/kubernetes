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

package v2

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/info/v1"
)

func machineFsStatsFromV1(fsStats []v1.FsStats) []MachineFsStats {
	var result []MachineFsStats
	for _, stat := range fsStats {
		readDuration := time.Millisecond * time.Duration(stat.ReadTime)
		writeDuration := time.Millisecond * time.Duration(stat.WriteTime)
		ioDuration := time.Millisecond * time.Duration(stat.IoTime)
		weightedDuration := time.Millisecond * time.Duration(stat.WeightedIoTime)
		result = append(result, MachineFsStats{
			Device:     stat.Device,
			Type:       stat.Type,
			Capacity:   &stat.Limit,
			Usage:      &stat.Usage,
			Available:  &stat.Available,
			InodesFree: &stat.InodesFree,
			DiskStats: DiskStats{
				ReadsCompleted:     &stat.ReadsCompleted,
				ReadsMerged:        &stat.ReadsMerged,
				SectorsRead:        &stat.SectorsRead,
				ReadDuration:       &readDuration,
				WritesCompleted:    &stat.WritesCompleted,
				WritesMerged:       &stat.WritesMerged,
				SectorsWritten:     &stat.SectorsWritten,
				WriteDuration:      &writeDuration,
				IoInProgress:       &stat.IoInProgress,
				IoDuration:         &ioDuration,
				WeightedIoDuration: &weightedDuration,
			},
		})
	}
	return result
}

func MachineStatsFromV1(cont *v1.ContainerInfo) []MachineStats {
	var stats []MachineStats
	var last *v1.ContainerStats
	for _, val := range cont.Stats {
		stat := MachineStats{
			Timestamp: val.Timestamp,
		}
		if cont.Spec.HasCpu {
			stat.Cpu = &val.Cpu
			cpuInst, err := InstCpuStats(last, val)
			if err != nil {
				glog.Warningf("Could not get instant cpu stats: %v", err)
			} else {
				stat.CpuInst = cpuInst
			}
			last = val
		}
		if cont.Spec.HasMemory {
			stat.Memory = &val.Memory
		}
		if cont.Spec.HasNetwork {
			stat.Network = &NetworkStats{
				// FIXME: Use reflection instead.
				Tcp:        TcpStat(val.Network.Tcp),
				Tcp6:       TcpStat(val.Network.Tcp6),
				Interfaces: val.Network.Interfaces,
			}
		}
		if cont.Spec.HasFilesystem {
			stat.Filesystem = machineFsStatsFromV1(val.Filesystem)
		}
		// TODO(rjnagal): Handle load stats.
		stats = append(stats, stat)
	}
	return stats
}

func ContainerStatsFromV1(spec *v1.ContainerSpec, stats []*v1.ContainerStats) []*ContainerStats {
	newStats := make([]*ContainerStats, 0, len(stats))
	var last *v1.ContainerStats
	for _, val := range stats {
		stat := &ContainerStats{
			Timestamp: val.Timestamp,
		}
		if spec.HasCpu {
			stat.Cpu = &val.Cpu
			cpuInst, err := InstCpuStats(last, val)
			if err != nil {
				glog.Warningf("Could not get instant cpu stats: %v", err)
			} else {
				stat.CpuInst = cpuInst
			}
			last = val
		}
		if spec.HasMemory {
			stat.Memory = &val.Memory
		}
		if spec.HasNetwork {
			// TODO: Handle TcpStats
			stat.Network = &NetworkStats{
				Interfaces: val.Network.Interfaces,
			}
		}
		if spec.HasFilesystem {
			if len(val.Filesystem) == 1 {
				stat.Filesystem = &FilesystemStats{
					TotalUsageBytes: &val.Filesystem[0].Usage,
					BaseUsageBytes:  &val.Filesystem[0].BaseUsage,
				}
			} else if len(val.Filesystem) > 1 {
				// Cannot handle multiple devices per container.
				glog.V(2).Infof("failed to handle multiple devices for container. Skipping Filesystem stats")
			}
		}
		if spec.HasDiskIo {
			stat.DiskIo = &val.DiskIo
		}
		if spec.HasCustomMetrics {
			stat.CustomMetrics = val.CustomMetrics
		}
		// TODO(rjnagal): Handle load stats.
		newStats = append(newStats, stat)
	}
	return newStats
}

func DeprecatedStatsFromV1(cont *v1.ContainerInfo) []DeprecatedContainerStats {
	stats := make([]DeprecatedContainerStats, 0, len(cont.Stats))
	var last *v1.ContainerStats
	for _, val := range cont.Stats {
		stat := DeprecatedContainerStats{
			Timestamp:        val.Timestamp,
			HasCpu:           cont.Spec.HasCpu,
			HasMemory:        cont.Spec.HasMemory,
			HasNetwork:       cont.Spec.HasNetwork,
			HasFilesystem:    cont.Spec.HasFilesystem,
			HasDiskIo:        cont.Spec.HasDiskIo,
			HasCustomMetrics: cont.Spec.HasCustomMetrics,
		}
		if stat.HasCpu {
			stat.Cpu = val.Cpu
			cpuInst, err := InstCpuStats(last, val)
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
		stats = append(stats, stat)
	}
	return stats
}

func InstCpuStats(last, cur *v1.ContainerStats) (*CpuInstStats, error) {
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
	return &CpuInstStats{
		Usage: CpuInstUsage{
			Total:  total,
			PerCpu: percpu,
			User:   user,
			System: system,
		},
	}, nil
}

// Get V2 container spec from v1 container info.
func ContainerSpecFromV1(specV1 *v1.ContainerSpec, aliases []string, namespace string) ContainerSpec {
	specV2 := ContainerSpec{
		CreationTime:     specV1.CreationTime,
		HasCpu:           specV1.HasCpu,
		HasMemory:        specV1.HasMemory,
		HasFilesystem:    specV1.HasFilesystem,
		HasNetwork:       specV1.HasNetwork,
		HasDiskIo:        specV1.HasDiskIo,
		HasCustomMetrics: specV1.HasCustomMetrics,
		Image:            specV1.Image,
		Labels:           specV1.Labels,
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

func instCpuStats(last, cur *v1.ContainerStats) (*CpuInstStats, error) {
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
	return &CpuInstStats{
		Usage: CpuInstUsage{
			Total:  total,
			PerCpu: percpu,
			User:   user,
			System: system,
		},
	}, nil
}
