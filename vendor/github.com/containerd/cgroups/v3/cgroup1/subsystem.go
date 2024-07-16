/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package cgroup1

import (
	"fmt"
	"os"

	"github.com/containerd/cgroups/v3"
	v1 "github.com/containerd/cgroups/v3/cgroup1/stats"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// Name is a typed name for a cgroup subsystem
type Name string

const (
	Devices   Name = "devices"
	Hugetlb   Name = "hugetlb"
	Freezer   Name = "freezer"
	Pids      Name = "pids"
	NetCLS    Name = "net_cls"
	NetPrio   Name = "net_prio"
	PerfEvent Name = "perf_event"
	Cpuset    Name = "cpuset"
	Cpu       Name = "cpu"
	Cpuacct   Name = "cpuacct"
	Memory    Name = "memory"
	Blkio     Name = "blkio"
	Rdma      Name = "rdma"
)

// Subsystems returns a complete list of the default cgroups
// available on most linux systems
func Subsystems() []Name {
	n := []Name{
		Freezer,
		Pids,
		NetCLS,
		NetPrio,
		PerfEvent,
		Cpuset,
		Cpu,
		Cpuacct,
		Memory,
		Blkio,
		Rdma,
	}
	if !cgroups.RunningInUserNS() {
		n = append(n, Devices)
	}
	if _, err := os.Stat("/sys/kernel/mm/hugepages"); err == nil {
		n = append(n, Hugetlb)
	}
	return n
}

type Subsystem interface {
	Name() Name
}

type pather interface {
	Subsystem
	Path(path string) string
}

type creator interface {
	Subsystem
	Create(path string, resources *specs.LinuxResources) error
}

type deleter interface {
	Subsystem
	Delete(path string) error
}

type stater interface {
	Subsystem
	Stat(path string, stats *v1.Metrics) error
}

type updater interface {
	Subsystem
	Update(path string, resources *specs.LinuxResources) error
}

// SingleSubsystem returns a single cgroup subsystem within the base Hierarchy
func SingleSubsystem(baseHierarchy Hierarchy, subsystem Name) Hierarchy {
	return func() ([]Subsystem, error) {
		subsystems, err := baseHierarchy()
		if err != nil {
			return nil, err
		}
		for _, s := range subsystems {
			if s.Name() == subsystem {
				return []Subsystem{
					s,
				}, nil
			}
		}
		return nil, fmt.Errorf("unable to find subsystem %s", subsystem)
	}
}
