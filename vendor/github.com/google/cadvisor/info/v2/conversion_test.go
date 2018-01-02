// Copyright 2016 Google Inc. All Rights Reserved.
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
	"reflect"
	"testing"
	"time"

	"github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
)

var (
	timestamp = time.Date(1987, time.August, 10, 0, 0, 0, 0, time.UTC)
	labels    = map[string]string{"foo": "bar"}
	envs      = map[string]string{"foo": "bar"}
)

func TestContanierSpecFromV1(t *testing.T) {
	v1Spec := v1.ContainerSpec{
		CreationTime: timestamp,
		Labels:       labels,
		Envs:         envs,
		HasCpu:       true,
		Cpu: v1.CpuSpec{
			Limit:    2048,
			MaxLimit: 4096,
			Mask:     "cpu_mask",
		},
		HasMemory: true,
		Memory: v1.MemorySpec{
			Limit:       2048,
			Reservation: 1024,
			SwapLimit:   8192,
		},
		HasNetwork:       true,
		HasFilesystem:    true,
		HasDiskIo:        true,
		HasCustomMetrics: true,
		CustomMetrics: []v1.MetricSpec{{
			Name:   "foo",
			Type:   v1.MetricGauge,
			Format: v1.IntType,
			Units:  "bars",
		}},
		Image: "gcr.io/kubernetes/kubernetes:v1",
	}

	aliases := []string{"baz", "oof"}
	namespace := "foo_bar_baz"

	expectedV2Spec := ContainerSpec{
		CreationTime: timestamp,
		Labels:       labels,
		Envs:         envs,
		HasCpu:       true,
		Cpu: CpuSpec{
			Limit:    2048,
			MaxLimit: 4096,
			Mask:     "cpu_mask",
		},
		HasMemory: true,
		Memory: MemorySpec{
			Limit:       2048,
			Reservation: 1024,
			SwapLimit:   8192,
		},
		HasNetwork:       true,
		HasFilesystem:    true,
		HasDiskIo:        true,
		HasCustomMetrics: true,
		CustomMetrics: []v1.MetricSpec{{
			Name:   "foo",
			Type:   v1.MetricGauge,
			Format: v1.IntType,
			Units:  "bars",
		}},
		Image:     "gcr.io/kubernetes/kubernetes:v1",
		Aliases:   aliases,
		Namespace: namespace,
	}

	v2Spec := ContainerSpecFromV1(&v1Spec, aliases, namespace)
	if !reflect.DeepEqual(v2Spec, expectedV2Spec) {
		t.Errorf("Converted spec differs from expectation!\nExpected: %+v\n Got: %+v\n", expectedV2Spec, v2Spec)
	}
}

func TestContainerStatsFromV1(t *testing.T) {
	v1Spec := v1.ContainerSpec{
		CreationTime: timestamp,
		Labels:       labels,
		HasCpu:       true,
		Cpu: v1.CpuSpec{
			Limit:    2048,
			MaxLimit: 4096,
			Mask:     "cpu_mask",
		},
		HasMemory: true,
		Memory: v1.MemorySpec{
			Limit:       2048,
			Reservation: 1024,
			SwapLimit:   8192,
		},
		HasNetwork:       true,
		HasFilesystem:    true,
		HasDiskIo:        true,
		HasCustomMetrics: true,
		CustomMetrics: []v1.MetricSpec{{
			Name:   "foo",
			Type:   v1.MetricGauge,
			Format: v1.IntType,
			Units:  "bars",
		}},
		Image: "gcr.io/kubernetes/kubernetes:v1",
	}
	v1Stats := v1.ContainerStats{
		Timestamp: timestamp,
		Memory: v1.MemoryStats{
			Usage:      1,
			Cache:      2,
			RSS:        3,
			WorkingSet: 4,
			Failcnt:    5,
			ContainerData: v1.MemoryStatsMemoryData{
				Pgfault:    1,
				Pgmajfault: 2,
			},
			HierarchicalData: v1.MemoryStatsMemoryData{
				Pgfault:    10,
				Pgmajfault: 20,
			},
		},
		Network: v1.NetworkStats{
			InterfaceStats: v1.InterfaceStats{
				Name:      "",
				RxBytes:   1,
				RxPackets: 2,
				RxErrors:  3,
				RxDropped: 4,
				TxBytes:   5,
				TxPackets: 6,
				TxErrors:  7,
				TxDropped: 8,
			},
			Interfaces: []v1.InterfaceStats{{
				Name:      "eth0",
				RxBytes:   10,
				RxPackets: 20,
				RxErrors:  30,
				RxDropped: 40,
				TxBytes:   50,
				TxPackets: 60,
				TxErrors:  70,
				TxDropped: 80,
			}},
		},
		Filesystem: []v1.FsStats{{
			Device:     "dev0",
			Limit:      500,
			Usage:      100,
			BaseUsage:  50,
			Available:  300,
			InodesFree: 100,
		}},
		Accelerators: []v1.AcceleratorStats{{
			Make:        "nvidia",
			Model:       "tesla-p100",
			ID:          "GPU-deadbeef-1234-5678-90ab-feedfacecafe",
			MemoryTotal: 20304050607,
			MemoryUsed:  2030405060,
			DutyCycle:   12,
		}},
	}
	expectedV2Stats := ContainerStats{
		Timestamp: timestamp,
		Cpu:       &v1Stats.Cpu,
		DiskIo:    &v1Stats.DiskIo,
		Memory:    &v1Stats.Memory,
		Network: &NetworkStats{
			Interfaces: v1Stats.Network.Interfaces,
		},
		Filesystem: &FilesystemStats{
			TotalUsageBytes: &v1Stats.Filesystem[0].Usage,
			BaseUsageBytes:  &v1Stats.Filesystem[0].BaseUsage,
			InodeUsage:      &v1Stats.Filesystem[0].Inodes,
		},
		Accelerators: v1Stats.Accelerators,
	}

	v2Stats := ContainerStatsFromV1("test", &v1Spec, []*v1.ContainerStats{&v1Stats})
	actualV2Stats := *v2Stats[0]

	if !reflect.DeepEqual(expectedV2Stats, actualV2Stats) {
		t.Errorf("Converted stats differs from expectation!\nExpected: %+v\n Got: %+v\n", expectedV2Stats, actualV2Stats)
	}
}

func TestInstCpuStats(t *testing.T) {
	tests := []struct {
		last *v1.ContainerStats
		cur  *v1.ContainerStats
		want *CpuInstStats
	}{
		// Last is missing
		{
			nil,
			&v1.ContainerStats{},
			nil,
		},
		// Goes back in time
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			nil,
		},
		// Zero time delta
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			nil,
		},
		// Different number of cpus
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						PerCpu: []uint64{100, 200},
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						PerCpu: []uint64{100, 200, 300},
					},
				},
			},
			nil,
		},
		// Stat numbers decrease
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  200,
						PerCpu: []uint64{100, 100},
						User:   150,
						System: 50,
					},
				},
			},
			nil,
		},
		// One second elapsed
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  500,
						PerCpu: []uint64{200, 300},
						User:   400,
						System: 100,
					},
				},
			},
			&CpuInstStats{
				Usage: CpuInstUsage{
					Total:  200,
					PerCpu: []uint64{100, 100},
					User:   150,
					System: 50,
				},
			},
		},
		// Two seconds elapsed
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(2 * time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  500,
						PerCpu: []uint64{200, 300},
						User:   400,
						System: 100,
					},
				},
			},
			&CpuInstStats{
				Usage: CpuInstUsage{
					Total:  100,
					PerCpu: []uint64{50, 50},
					User:   75,
					System: 25,
				},
			},
		},
	}
	for _, c := range tests {
		got, err := InstCpuStats(c.last, c.cur)
		if err != nil {
			if c.want == nil {
				continue
			}
			t.Errorf("Unexpected error: %v", err)
		}
		assert.Equal(t, c.want, got)
	}
}
