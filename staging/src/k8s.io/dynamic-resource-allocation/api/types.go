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

type ResourceSlice struct {
	metav1.TypeMeta
	metav1.ObjectMeta
	Spec ResourceSliceSpec
}

type ResourceSliceSpec struct {
	Driver                 UniqueString
	Pool                   ResourcePool
	NodeName               *string
	NodeSelector           *v1.NodeSelector
	AllNodes               bool
	Devices                []Device
	PerDeviceNodeSelection *bool
	SharedCounters         []CounterSet
}

type CounterSet struct {
	Name     UniqueString
	Counters map[string]resourceapi.Counter
}

type ResourcePool struct {
	Name               UniqueString
	Generation         int64
	ResourceSliceCount int64
}
type Device struct {
	Name                     UniqueString
	Attributes               map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
	Capacity                 map[resourceapi.QualifiedName]resourceapi.DeviceCapacity
	ConsumesCounters         []DeviceCounterConsumption
	NodeName                 *string
	NodeSelector             *v1.NodeSelector
	AllNodes                 *bool
	Taints                   []resourceapi.DeviceTaint
	BindsToNode              bool
	BindingConditions        []string
	BindingFailureConditions []string
	AllowMultipleAllocations *bool
}

type DeviceCounterConsumption struct {
	CounterSet UniqueString
	Counters   map[string]resourceapi.Counter
}
