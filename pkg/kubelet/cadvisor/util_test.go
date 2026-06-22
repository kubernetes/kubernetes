//go:build cgo && linux

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
	"strings"
	"testing"

	"github.com/google/cadvisor/container/crio"
	info "github.com/google/cadvisor/info/v1"
	infov2 "github.com/google/cadvisor/info/v2"
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

// TestEphemeralStorageCapacityFromFsInfo verifies that capacity uses DecimalSI.
// Allocatable inherits capacity's format via DeepCopy().Sub() in setters.go.
// DecimalSI produces clean suffixed values for any byte count, while BinarySI
// falls back to bare integers for non-1024-aligned sizes. See #133927.
func TestEphemeralStorageCapacityFromFsInfo(t *testing.T) {
	// Use a non-1024-aligned capacity typical of real filesystem sizes.
	fsInfo := infov2.FsInfo{Capacity: 97842800000}
	capacity := EphemeralStorageCapacityFromFsInfo(fsInfo)
	ephemeralCapacity := capacity[v1.ResourceEphemeralStorage]

	if ephemeralCapacity.Format != resource.DecimalSI {
		t.Errorf("capacity format = %v, want DecimalSI", ephemeralCapacity.Format)
	}

	// Simulate allocatable derivation (setters.go): allocatable = capacity - reservation.
	// DeepCopy preserves the format, so allocatable inherits DecimalSI from capacity.
	allocatable := ephemeralCapacity.DeepCopy()
	reservation := resource.MustParse("1Gi")
	allocatable.Sub(reservation)

	if allocatable.Format != resource.DecimalSI {
		t.Errorf("allocatable format = %v, want DecimalSI (inherited from capacity)", allocatable.Format)
	}
}

func TestCrioSocket(t *testing.T) {
	assert.True(t, strings.HasSuffix(crio.CrioSocket, CrioSocketSuffix), "CrioSocketSuffix in this package must be a suffix of the one in github.com/google/cadvisor/container/crio/client.go")
}
