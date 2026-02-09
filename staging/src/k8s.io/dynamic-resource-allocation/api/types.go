/*
Copyright 2024 The Kubernetes Authors.

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

package api

import (
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// JSON tags exist to make the output more readable (klog, diff.Diff).
// They are intentionally not compatible with the normal encoding
// of a ResourceSlice to avoid accidentally using them with an apiserver
// request:
// - TypeMeta does not get encoded.
// - Fields from this package use upper case whereas types from the
//   real API use lower case.

type ResourceSlice struct {
	metav1.TypeMeta `json:"-"` // Not needed, not set consistently.
	metav1.ObjectMeta

	Spec ResourceSliceSpec
}

type ResourceSliceSpec struct {
	Driver                 UniqueString
	Pool                   ResourcePool
	NodeName               *string          `json:",omitempty"`
	NodeSelector           *v1.NodeSelector `json:",omitempty"`
	AllNodes               bool             `json:",omitempty"`
	Devices                []Device         `json:",omitempty"`
	PerDeviceNodeSelection *bool            `json:",omitempty"`
	SharedCounters         []CounterSet     `json:",omitempty"`
}

type CounterSet struct {
	Name     UniqueString
	Counters map[string]resourceapi.Counter `json:",omitempty"`
}

type ResourcePool struct {
	Name               UniqueString
	Generation         int64
	ResourceSliceCount int64
}

type Device struct {
	Name                     UniqueString
	Attributes               map[resourceapi.QualifiedName]resourceapi.DeviceAttribute `json:",omitempty"`
	Capacity                 map[resourceapi.QualifiedName]resourceapi.DeviceCapacity  `json:",omitempty"`
	ConsumesCounters         []DeviceCounterConsumption                                `json:",omitempty"`
	NodeName                 *string                                                   `json:",omitempty"`
	NodeSelector             *v1.NodeSelector                                          `json:",omitempty"`
	AllNodes                 *bool                                                     `json:",omitempty"`
	Taints                   []resourceapi.DeviceTaint                                 `json:",omitempty"`
	BindsToNode              bool                                                      `json:",omitempty"`
	BindingConditions        []string                                                  `json:",omitempty"`
	BindingFailureConditions []string                                                  `json:",omitempty"`
	AllowMultipleAllocations *bool                                                     `json:",omitempty"`
}

type DeviceCounterConsumption struct {
	CounterSet UniqueString
	Counters   map[string]resourceapi.Counter `json:",omitempty"`
}
