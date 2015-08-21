/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package v1

import (
	"time"

	"k8s.io/kubernetes/pkg/api/resource"
)

type NodeStats struct {
	TypeMeta
	ObjectMeta // should have node name
	Containers []ContainerStats
}

type ContainerStats struct {
	Name  string        `json:"name"`
	Stats []*StatsPoint `json:"stats"`
}

type StatsPoint struct {
	Time time.Time `json:"time"`

	Cpu    CpuStats    `json:"cpu"`
	Memory MemoryStats `json:"memory"`
}

type CpuStats struct {
	// Current CPU usage
	// Unit: Millicores per second
	Usage resource.Quantity `json:"usage"`
}

type MemoryStats struct {
	// Current memory usage, this includes all memory regardless of when it was
	// accessed.
	// Unit: Bytes
	Usage resource.Quantity `json:"usage"`

	// The amount of working set memory, this includes recently accessed memory,
	// dirty memory, and kernel memory. Working set is <= "usage".
	// Unit: Bytes
	WorkingSet resource.Quantity `json:"working_set"`
}
