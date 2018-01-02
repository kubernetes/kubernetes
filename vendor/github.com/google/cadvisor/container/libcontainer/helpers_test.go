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

package libcontainer

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	info "github.com/google/cadvisor/info/v1"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/system"
)

func TestScanInterfaceStats(t *testing.T) {
	stats, err := scanInterfaceStats("testdata/procnetdev")
	if err != nil {
		t.Error(err)
	}

	var netdevstats = []info.InterfaceStats{
		{
			Name:      "wlp4s0",
			RxBytes:   1,
			RxPackets: 2,
			RxErrors:  3,
			RxDropped: 4,
			TxBytes:   9,
			TxPackets: 10,
			TxErrors:  11,
			TxDropped: 12,
		},
		{
			Name:      "em1",
			RxBytes:   315849,
			RxPackets: 1172,
			RxErrors:  0,
			RxDropped: 0,
			TxBytes:   315850,
			TxPackets: 1173,
			TxErrors:  0,
			TxDropped: 0,
		},
	}

	if len(stats) != len(netdevstats) {
		t.Errorf("Expected 2 net stats, got %d", len(stats))
	}

	for i, v := range netdevstats {
		if v != stats[i] {
			t.Errorf("Expected %#v, got %#v", v, stats[i])
		}
	}
}

func TestScanUDPStats(t *testing.T) {
	udpStatsFile := "testdata/procnetudp"
	r, err := os.Open(udpStatsFile)
	if err != nil {
		t.Errorf("failure opening %s: %v", udpStatsFile, err)
	}

	stats, err := scanUdpStats(r)
	if err != nil {
		t.Error(err)
	}

	var udpstats = info.UdpStat{
		Listen:   2,
		Dropped:  4,
		RxQueued: 10,
		TxQueued: 11,
	}

	if stats != udpstats {
		t.Errorf("Expected %#v, got %#v", udpstats, stats)
	}
}

// https://github.com/docker/libcontainer/blob/v2.2.1/cgroups/fs/cpuacct.go#L19
const nanosecondsInSeconds = 1000000000

var clockTicks = uint64(system.GetClockTicks())

func TestMorePossibleCPUs(t *testing.T) {
	realNumCPUs := uint32(8)
	numCpusFunc = func() (uint32, error) {
		return realNumCPUs, nil
	}
	possibleCPUs := uint32(31)

	perCpuUsage := make([]uint64, possibleCPUs)
	for i := uint32(0); i < realNumCPUs; i++ {
		perCpuUsage[i] = 8562955455524
	}

	s := &cgroups.Stats{
		CpuStats: cgroups.CpuStats{
			CpuUsage: cgroups.CpuUsage{
				PercpuUsage:       perCpuUsage,
				TotalUsage:        33802947350272,
				UsageInKernelmode: 734746 * nanosecondsInSeconds / clockTicks,
				UsageInUsermode:   2767637 * nanosecondsInSeconds / clockTicks,
			},
		},
	}
	var ret info.ContainerStats
	setCpuStats(s, &ret)

	expected := info.ContainerStats{
		Cpu: info.CpuStats{
			Usage: info.CpuUsage{
				PerCpu: perCpuUsage[0:realNumCPUs],
				User:   s.CpuStats.CpuUsage.UsageInUsermode,
				System: s.CpuStats.CpuUsage.UsageInKernelmode,
				Total:  8562955455524 * uint64(realNumCPUs),
			},
		},
	}

	if !ret.Eq(&expected) {
		t.Fatalf("expected %+v == %+v", ret, expected)
	}
}

var defaultCgroupSubsystems = []string{
	"systemd", "freezer", "memory", "blkio", "hugetlb", "net_cls,net_prio", "pids", "cpu,cpuacct", "devices", "cpuset", "perf_events",
}

func cgroupMountsAt(path string, subsystems []string) []cgroups.Mount {
	res := []cgroups.Mount{}
	for _, subsystem := range subsystems {
		res = append(res, cgroups.Mount{
			Root:       "/",
			Subsystems: strings.Split(subsystem, ","),
			Mountpoint: filepath.Join(path, subsystem),
		})
	}
	return res
}

func TestGetCgroupSubsystems(t *testing.T) {
	ourSubsystems := []string{"cpu,cpuacct", "devices", "memory", "cpuset", "blkio"}

	testCases := []struct {
		mounts   []cgroups.Mount
		expected CgroupSubsystems
		err      bool
	}{
		{
			mounts: []cgroups.Mount{},
			err:    true,
		},
		{
			// normal case
			mounts: cgroupMountsAt("/sys/fs/cgroup", defaultCgroupSubsystems),
			expected: CgroupSubsystems{
				MountPoints: map[string]string{
					"blkio":   "/sys/fs/cgroup/blkio",
					"cpu":     "/sys/fs/cgroup/cpu,cpuacct",
					"cpuacct": "/sys/fs/cgroup/cpu,cpuacct",
					"cpuset":  "/sys/fs/cgroup/cpuset",
					"devices": "/sys/fs/cgroup/devices",
					"memory":  "/sys/fs/cgroup/memory",
				},
				Mounts: cgroupMountsAt("/sys/fs/cgroup", ourSubsystems),
			},
		},
		{
			// multiple croup subsystems, should ignore second one
			mounts: append(cgroupMountsAt("/sys/fs/cgroup", defaultCgroupSubsystems),
				cgroupMountsAt("/var/lib/rkt/pods/run/ccdd4e36-2d4c-49fd-8b94-4fb06133913d/stage1/rootfs/opt/stage2/flannel/rootfs/sys/fs/cgroup", defaultCgroupSubsystems)...),
			expected: CgroupSubsystems{
				MountPoints: map[string]string{
					"blkio":   "/sys/fs/cgroup/blkio",
					"cpu":     "/sys/fs/cgroup/cpu,cpuacct",
					"cpuacct": "/sys/fs/cgroup/cpu,cpuacct",
					"cpuset":  "/sys/fs/cgroup/cpuset",
					"devices": "/sys/fs/cgroup/devices",
					"memory":  "/sys/fs/cgroup/memory",
				},
				Mounts: cgroupMountsAt("/sys/fs/cgroup", ourSubsystems),
			},
		},
		{
			// most subsystems not mounted
			mounts: append(cgroupMountsAt("/sys/fs/cgroup", []string{"cpu"})),
			expected: CgroupSubsystems{
				MountPoints: map[string]string{
					"cpu": "/sys/fs/cgroup/cpu",
				},
				Mounts: cgroupMountsAt("/sys/fs/cgroup", []string{"cpu"}),
			},
		},
	}

	for i, testCase := range testCases {
		subSystems, err := getCgroupSubsystemsHelper(testCase.mounts)
		if testCase.err {
			if err == nil {
				t.Fatalf("[case %d] Expected error but didn't get one", i)
			}
			continue
		}
		if err != nil {
			t.Fatalf("[case %d] Expected no error, but got %v", i, err)
		}
		assertCgroupSubsystemsEqual(t, testCase.expected, subSystems, fmt.Sprintf("[case %d]", i))
	}
}

func assertCgroupSubsystemsEqual(t *testing.T, expected, actual CgroupSubsystems, message string) {
	if !reflect.DeepEqual(expected.MountPoints, actual.MountPoints) {
		t.Fatalf("%s Expected %v == %v", message, expected.MountPoints, actual.MountPoints)
	}

	sort.Slice(expected.Mounts, func(i, j int) bool {
		return expected.Mounts[i].Mountpoint < expected.Mounts[j].Mountpoint
	})
	sort.Slice(actual.Mounts, func(i, j int) bool {
		return actual.Mounts[i].Mountpoint < actual.Mounts[j].Mountpoint
	})
	if !reflect.DeepEqual(expected.Mounts, actual.Mounts) {
		t.Fatalf("%s Expected %v == %v", message, expected.Mounts, actual.Mounts)
	}
}
