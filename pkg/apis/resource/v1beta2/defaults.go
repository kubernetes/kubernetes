/*
Copyright 2022 The Kubernetes Authors.

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

package v1beta2

import (
	"time"

	resourceapi "k8s.io/api/resource/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_DeviceRequest(obj *resourceapi.DeviceRequest) {
	for i := range obj.FirstAvailable {
		if obj.FirstAvailable[i].AllocationMode == "" {
			obj.FirstAvailable[i].AllocationMode = resourceapi.DeviceAllocationModeExactCount
		}

		// We allow the count to be zero for the last entry in the FirstAvailable list
		// unless it is the only subrequest. So for other situations, we default the Count
		// to 1 if it set to zero and the allocationMode is ExactCount.
		if len(obj.FirstAvailable) == 1 || i < len(obj.FirstAvailable)-1 {
			if obj.FirstAvailable[i].AllocationMode == resourceapi.DeviceAllocationModeExactCount && obj.FirstAvailable[i].Count == 0 {
				obj.FirstAvailable[i].Count = 1
			}
		}
	}
}

func SetDefaults_ExactDeviceRequest(obj *resourceapi.ExactDeviceRequest) {
	if obj.AllocationMode == "" {
		obj.AllocationMode = resourceapi.DeviceAllocationModeExactCount
	}

	if obj.AllocationMode == resourceapi.DeviceAllocationModeExactCount && obj.Count == 0 {
		obj.Count = 1
	}
}

func SetDefaults_DeviceTaint(obj *resourceapi.DeviceTaint) {
	if obj.TimeAdded == nil {
		obj.TimeAdded = &metav1.Time{Time: time.Now().Truncate(time.Second)}
	}
}
