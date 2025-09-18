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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

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
	Devices                []SliceDevice // Manual conversion maps this to the public []resourceapi.Device.
	Taints                 []SliceDeviceTaint
	PerDeviceNodeSelection *bool
	SharedCounters         []CounterSet
}

type CounterSet struct {
	Name     UniqueString
	Counters map[string]Counter
}

type ResourcePool struct {
	Name               UniqueString
	Generation         int64
	ResourceSliceCount int64
}

// SliceDevice is an extension of the public type with
// additional fields which only get set after conversion.
type SliceDevice struct {
	Device
	Taints []resourceapi.DeviceTaint
}

type Device struct {
	Name                     UniqueString
	Attributes               map[QualifiedName]DeviceAttribute
	Capacity                 map[QualifiedName]DeviceCapacity
	ConsumesCounters         []DeviceCounterConsumption
	NodeName                 *string
	NodeSelector             *v1.NodeSelector
	AllNodes                 *bool
	BindsToNode              bool
	BindingConditions        []string
	BindingFailureConditions []string
	AllowMultipleAllocations *bool
}

type DeviceCounterConsumption struct {
	CounterSet UniqueString
	Counters   map[string]Counter
}

type QualifiedName string

type FullyQualifiedName string

type DeviceAttribute struct {
	IntValue     *int64
	BoolValue    *bool
	StringValue  *string
	VersionValue *string
}

type DeviceCapacity struct {
	Value         resource.Quantity
	RequestPolicy *CapacityRequestPolicy
}

type CapacityRequestPolicy struct {
	Default     *resource.Quantity
	ValidValues []resource.Quantity
	ValidRange  *CapacityRequestPolicyRange
}

type CapacityRequestPolicyRange struct {
	Min  *resource.Quantity
	Max  *resource.Quantity
	Step *resource.Quantity
}

type Counter struct {
	Value resource.Quantity
}

type SliceDeviceTaint struct {
	Device UniqueString
	resourceapi.DeviceTaint
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type ResourceSliceList struct {
	metav1.TypeMeta
	metav1.ListMeta
	Items []ResourceSlice
}
