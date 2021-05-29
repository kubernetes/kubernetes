// +build libpfm,cgo

// Copyright 2020 Google Inc. All Rights Reserved.
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

// Manager of perf events for containers.
package perf

import (
	"fmt"
	"os"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/stats"
	"github.com/google/cadvisor/utils/sysinfo"
)

type manager struct {
	events      PerfEvents
	onlineCPUs  []int
	cpuToSocket map[int]int
	stats.NoopDestroy
}

func NewManager(configFile string, topology []info.Node) (stats.Manager, error) {
	if configFile == "" {
		return &stats.NoopManager{}, nil
	}

	file, err := os.Open(configFile)
	if err != nil {
		return nil, fmt.Errorf("unable to read configuration file %q: %w", configFile, err)
	}

	config, err := parseConfig(file)
	if err != nil {
		return nil, fmt.Errorf("unable to parse configuration file %q: %w", configFile, err)
	}

	onlineCPUs := sysinfo.GetOnlineCPUs(topology)

	cpuToSocket := make(map[int]int)

	for _, cpu := range onlineCPUs {
		cpuToSocket[cpu] = sysinfo.GetSocketFromCPU(topology, cpu)
	}

	return &manager{events: config, onlineCPUs: onlineCPUs, cpuToSocket: cpuToSocket}, nil
}

func (m *manager) GetCollector(cgroupPath string) (stats.Collector, error) {
	collector := newCollector(cgroupPath, m.events, m.onlineCPUs, m.cpuToSocket)
	err := collector.setup()
	if err != nil {
		collector.Destroy()
		return &stats.NoopCollector{}, err
	}
	return collector, nil
}
