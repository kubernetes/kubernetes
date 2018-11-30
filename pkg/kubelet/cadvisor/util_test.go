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
	"testing"

	info "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
)

func TestCapacityFromMachineInfo(t *testing.T) {
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

	// enable the features.HugePages
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HugePages, true)()

	resourceList := CapacityFromMachineInfo(machineInfo)

	// assert the cpu and memory
	assert.EqualValues(t, machineInfo.NumCores*1000, resourceList.Cpu().MilliValue(), "unexpected CPU value")
	assert.EqualValues(t, machineInfo.MemoryCapacity, resourceList.Memory().Value(), "unexpected memory value")

	// assert the hugepage
	hugePageKey := int64(5 * 1024)
	value, found := resourceList[v1helper.HugePageResourceName(*resource.NewQuantity(hugePageKey, resource.BinarySI))]
	assert.True(t, found, "hugepage not found")
	assert.EqualValues(t, hugePageKey*10, value.Value(), "unexpected hugepage value")
}
