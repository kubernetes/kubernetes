/*
Copyright 2017 The Kubernetes Authors.

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

package cpumanager

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"testing"
)

// CpuAllocatable must be <= CpuCapacity
func prepareCPUNodeStatus(CPUCapacity, CPUAllocatable string) v1.NodeStatus {
	nodestatus := v1.NodeStatus{
		Capacity:    make(v1.ResourceList, 1),
		Allocatable: make(v1.ResourceList, 1),
	}
	cpucap, _ := resource.ParseQuantity(CPUCapacity)
	cpuall, _ := resource.ParseQuantity(CPUAllocatable)

	nodestatus.Capacity[v1.ResourceCPU] = cpucap
	nodestatus.Allocatable[v1.ResourceCPU] = cpuall
	return nodestatus
}

func TestGetReservedCpus(t *testing.T) {
	var reservedCPUTests = []struct {
		cpuCapacity     string
		cpuAllocatable  string
		expReservedCpus int
	}{
		{cpuCapacity: "1000m", cpuAllocatable: "1000m", expReservedCpus: 0},
		{cpuCapacity: "8000m", cpuAllocatable: "7500m", expReservedCpus: 0},
		{cpuCapacity: "16000m", cpuAllocatable: "14100m", expReservedCpus: 1},
		{cpuCapacity: "8000m", cpuAllocatable: "5500m", expReservedCpus: 2},
		{cpuCapacity: "8000m", cpuAllocatable: "900m", expReservedCpus: 7},
	}

	for idx, test := range reservedCPUTests {
		tmpNodeStaus := prepareCPUNodeStatus(test.cpuCapacity, test.cpuAllocatable)
		gotReservedCpus := getReserverdCpus(tmpNodeStaus)
		if test.expReservedCpus != gotReservedCpus {
			t.Errorf("(Case %d) Expected reserved cpus %d, got %d",
				idx, test.expReservedCpus, gotReservedCpus)
		}
	}
}
