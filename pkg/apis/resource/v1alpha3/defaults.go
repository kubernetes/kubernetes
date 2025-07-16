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

package v1alpha3

import (
	"time"

	resourceapi "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_DeviceRequest(obj *resourceapi.DeviceRequest) {
	// If the deviceClassName is not set, then the request will have
	// subrequests and the allocationMode and count fields should not
	// be set.
	if obj.DeviceClassName == "" {
		return
	}
	if obj.AllocationMode == "" {
		obj.AllocationMode = resourceapi.DeviceAllocationModeExactCount
	}

	if obj.AllocationMode == resourceapi.DeviceAllocationModeExactCount && obj.Count == 0 {
		obj.Count = 1
	}
}

func SetDefaults_DeviceSubRequest(obj *resourceapi.DeviceSubRequest) {
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
