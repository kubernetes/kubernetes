// +build cgo,linux

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

package cadvisor

import (
	"reflect"
	"testing"

	cadvisormetrics "github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/crio"
	info "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestCapacityFromMachineInfoWithHugePagesEnable(t *testing.T) {
	machineInfo := &info.MachineInfo{
		NumCores:       2,
		MemoryCapacity: 2048,
		HugePages: []info.HugePagesInfo{
			{
				PageSize: 5,
				NumPages: 10,
			},
		},
	}

	expected := v1.ResourceList{
		v1.ResourceCPU:    *resource.NewMilliQuantity(int64(2000), resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(int64(2048), resource.BinarySI),
		"hugepages-5Ki":   *resource.NewQuantity(int64(51200), resource.BinarySI),
	}
	actual := CapacityFromMachineInfo(machineInfo)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("when set hugepages true, got resource list %v, want %v", actual, expected)
	}
}

func TestCrioSocket(t *testing.T) {
	assert.EqualValues(t, CrioSocket, crio.CrioSocket, "CrioSocket in this package must equal the one in github.com/google/cadvisor/container/crio/client.go")
}

func TestDefaultCadvisorMetricSet(t *testing.T) {
	usingLegacyStatsTable := []bool{false, true}
	usingLocalStorageCapacityIsolationTable := []bool{false, false}
	disableAcceleratorUsageMetricsTable := []bool{true, false}

	metricSetWithNoDiskUsageAndNoAccelerator := cadvisormetrics.MetricSet{}
	for metricKind := range IncludedMetrics {
		metricSetWithNoDiskUsageAndNoAccelerator.Add(metricKind)
	}
	metricSetWithDiskUsageAndAccelerator := cadvisormetrics.MetricSet{}
	for metricKind := range IncludedMetrics {
		metricSetWithDiskUsageAndAccelerator.Add(metricKind)
	}
	metricSetWithDiskUsageAndAccelerator.Add(cadvisormetrics.DiskUsageMetrics)
	metricSetWithDiskUsageAndAccelerator.Add(cadvisormetrics.AcceleratorUsageMetrics)

	expectedTable := []cadvisormetrics.MetricSet{metricSetWithNoDiskUsageAndNoAccelerator, metricSetWithDiskUsageAndAccelerator}

	for idx := 0; idx < len(usingLegacyStatsTable); idx++ {
		actual := DefaultCadvisorMetricSet(usingLegacyStatsTable[idx], usingLocalStorageCapacityIsolationTable[idx], disableAcceleratorUsageMetricsTable[idx])
		if !reflect.DeepEqual(actual, expectedTable[idx]) {
			t.Errorf("test default metric set for cadvisor error, expected: %+v, actual: %+v", expectedTable[idx], actual)
		}
	}
}

func TestCustomCadvisorMetricSet(t *testing.T) {
	usingLegacyStatsTable := []bool{false, true}
	usingLocalStorageCapacityIsolationTable := []bool{false, false}
	disableAcceleratorUsageMetricsTable := []bool{true, false}
	customMetricSet := []string{"cpu", "memory", "disk", "illegal"}

	metricSet1 := cadvisormetrics.MetricSet{
		cadvisormetrics.CpuUsageMetrics:    struct{}{},
		cadvisormetrics.MemoryUsageMetrics: struct{}{},
	}
	metricSet2 := cadvisormetrics.MetricSet{
		cadvisormetrics.CpuUsageMetrics:    struct{}{},
		cadvisormetrics.MemoryUsageMetrics: struct{}{},
		cadvisormetrics.DiskUsageMetrics:   struct{}{},
	}
	expectedTable := []cadvisormetrics.MetricSet{metricSet1, metricSet2}

	for idx := 0; idx < len(usingLegacyStatsTable); idx++ {
		actual := CustomCadvisorMetricSet(usingLegacyStatsTable[idx], usingLocalStorageCapacityIsolationTable[idx], disableAcceleratorUsageMetricsTable[idx], customMetricSet)
		if !reflect.DeepEqual(actual, expectedTable[idx]) {
			t.Errorf("test custom metric set for cadvisor error, expected: %+v, actual: %+v", expectedTable[idx], actual)
		}
	}
}
