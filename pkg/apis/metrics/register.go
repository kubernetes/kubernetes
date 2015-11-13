/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package metrics

import (
	"k8s.io/kubernetes/pkg/api"
)

func init() {
	// Register the API.
	addKnownTypes()
}

// Adds the list of known types to api.Scheme.
func addKnownTypes() {
	api.Scheme.AddKnownTypes("",
		&RawNode{},
		&RawPod{},
		&RawContainer{},
		&ContainerSpec{},
		&CpuSpec{},
		&MemorySpec{},
		&ContainerStats{},
		&TcpStat{},
		&NetworkStats{},
		&InterfaceStats{},
		&CpuInstStats{},
		&CpuInstUsage{},
		&CpuUsage{},
		&CpuStats{},
		&PerDiskStats{},
		&DiskIoStats{},
		&MemoryStats{},
		&MemoryStatsMemoryData{},
		&FsStats{},
		&LoadStats{},
		&MetricSpec{},
		&CustomMetric{},
		&MetricVal{},
	)
}

func (*RawNode) IsAnAPIObject()               {}
func (*RawPod) IsAnAPIObject()                {}
func (*RawContainer) IsAnAPIObject()          {}
func (*ContainerSpec) IsAnAPIObject()         {}
func (*CpuSpec) IsAnAPIObject()               {}
func (*MemorySpec) IsAnAPIObject()            {}
func (*ContainerStats) IsAnAPIObject()        {}
func (*TcpStat) IsAnAPIObject()               {}
func (*NetworkStats) IsAnAPIObject()          {}
func (*InterfaceStats) IsAnAPIObject()        {}
func (*CpuInstStats) IsAnAPIObject()          {}
func (*CpuInstUsage) IsAnAPIObject()          {}
func (*CpuUsage) IsAnAPIObject()              {}
func (*CpuStats) IsAnAPIObject()              {}
func (*PerDiskStats) IsAnAPIObject()          {}
func (*DiskIoStats) IsAnAPIObject()           {}
func (*MemoryStats) IsAnAPIObject()           {}
func (*MemoryStatsMemoryData) IsAnAPIObject() {}
func (*FsStats) IsAnAPIObject()               {}
func (*LoadStats) IsAnAPIObject()             {}
func (*MetricSpec) IsAnAPIObject()            {}
func (*CustomMetric) IsAnAPIObject()          {}
func (*MetricVal) IsAnAPIObject()             {}
