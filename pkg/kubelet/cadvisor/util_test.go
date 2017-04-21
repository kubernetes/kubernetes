// +build cgo,linux

/*
Copyright 2016 The Kubernetes Authors.

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

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestCapacityFromMachineInfo(t *testing.T) {
	info := &cadvisorapi.MachineInfo{}
	info.NumCores = 5
	info.MemoryCapacity = 2048
	result := CapacityFromMachineInfo(info)
	expected := v1.ResourceList{
		v1.ResourceCPU: *resource.NewMilliQuantity(
			int64(5000),
			resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(
			int64(2048),
			resource.BinarySI),
	}

	if !reflect.DeepEqual(expected, result) {
		t.Errorf("expected %v, result %v", expected, result)
	}
}
