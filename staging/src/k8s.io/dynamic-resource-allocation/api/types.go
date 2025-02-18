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
	"k8s.io/apimachinery/pkg/api/resource"
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
	NodeName               UniqueString
	NodeSelector           *v1.NodeSelector
	AllNodes               bool
	Devices                []Device
	PerDeviceNodeSelection bool
	CapacityPools          []CapacityPool
	Mixins                 *ResourceSliceMixins
}

type ResourcePool struct {
	Name               UniqueString
	Generation         int64
	ResourceSliceCount int64
}

type CapacityPool struct {
	Name     UniqueString
	Includes []CapacityPoolMixinRef
	Capacity map[QualifiedName]DeviceCapacity
}

type CapacityPoolMixinRef struct {
	Name UniqueString
}

type ResourceSliceMixins struct {
	Device                    []DeviceMixin
	DeviceCapacityConsumption []DeviceCapacityConsumptionMixin
	CapacityPool              []CapacityPoolMixin
}

type DeviceCapacityConsumptionMixin struct {
	Name     UniqueString
	Capacity map[QualifiedName]DeviceCapacity
}

type CapacityPoolMixin struct {
	Name     UniqueString
	Capacity map[QualifiedName]DeviceCapacity
}

type DeviceMixin struct {
	Name      UniqueString
	Composite *CompositeDeviceMixin
}

type CompositeDeviceMixin struct {
	Attributes map[QualifiedName]DeviceAttribute
	Capacity   map[QualifiedName]DeviceCapacity
}

type Device struct {
	Name      UniqueString
	Basic     *BasicDevice
	Composite *CompositeDevice
}

type BasicDevice struct {
	Attributes map[QualifiedName]DeviceAttribute
	Capacity   map[QualifiedName]DeviceCapacity
}

type CompositeDevice struct {
	Includes         []DeviceMixinRef
	Attributes       map[QualifiedName]DeviceAttribute
	Capacity         map[QualifiedName]DeviceCapacity
	ConsumesCapacity []DeviceCapacityConsumption
	NodeName         string
	NodeSelector     *v1.NodeSelector
	AllNodes         bool
}

type DeviceMixinRef struct {
	Name UniqueString
}

type DeviceCapacityConsumption struct {
	CapacityPool string
	Includes     []DeviceCapacityConsumptionMixinRef
	Capacity     map[QualifiedName]DeviceCapacity
}

type DeviceCapacityConsumptionMixinRef struct {
	Name UniqueString
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
	Value resource.Quantity
}
