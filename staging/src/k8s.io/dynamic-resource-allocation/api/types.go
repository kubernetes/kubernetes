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

type ResourceSlice struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec ResourceSliceSpec `json:"spec"`
}

type ResourceSliceSpec struct {
	Driver                 UniqueString     `json:"driver"`
	Pool                   ResourcePool     `json:"pool"`
	NodeName               *string          `json:"nodeName,omitempty"`
	NodeSelector           *v1.NodeSelector `json:"nodeSelector,omitempty"`
	AllNodes               bool             `json:"allNodes,omitempty"`
	Devices                []Device         `json:"devices,omitempty"`
	PerDeviceNodeSelection *bool            `json:"perDeviceNodeSelection,omitempty"`
	SharedCounters         []CounterSet     `json:"sharedCounters,omitempty"`
}

type CounterSet struct {
	Name     UniqueString       `json:"name"`
	Counters map[string]Counter `json:"counters,omitempty"`
}

type ResourcePool struct {
	Name               UniqueString `json:"name"`
	Generation         int64        `json:"generation"`
	ResourceSliceCount int64        `json:"resourceSliceCount"`
}

type Device struct {
	Name                     UniqueString                      `json:"name"`
	Attributes               map[QualifiedName]DeviceAttribute `json:"attributes,omitempty"`
	Capacity                 map[QualifiedName]DeviceCapacity  `json:"capacity,omitempty"`
	ConsumesCounters         []DeviceCounterConsumption        `json:"consumesCounters,omitempty"`
	NodeName                 *string                           `json:"nodeName,omitempty"`
	NodeSelector             *v1.NodeSelector                  `json:"nodeSelector,omitempty"`
	AllNodes                 *bool                             `json:"allNodes,omitempty"`
	Taints                   []resourceapi.DeviceTaint         `json:"taints,omitempty"`
	BindsToNode              bool                              `json:"bindsToNode,omitempty"`
	BindingConditions        []string                          `json:"bindingConditions,omitempty"`
	BindingFailureConditions []string                          `json:"bindingFailureConditions,omitempty"`
	AllowMultipleAllocations *bool                             `json:"allowMultipleAllocations,omitempty"`
}

type DeviceCounterConsumption struct {
	CounterSet UniqueString       `json:"counterSet,omitempty"`
	Counters   map[string]Counter `json:"counters,omitempty"`
}

type QualifiedName string

type FullyQualifiedName string

type DeviceAttribute struct {
	IntValue     *int64  `json:"intValue,omitempty"`
	BoolValue    *bool   `json:"boolValue,omitempty"`
	StringValue  *string `json:"stringValue,omitempty"`
	VersionValue *string `json:"versionValue,omitempty"`
}

type DeviceCapacity struct {
	Value         resource.Quantity      `json:"value,omitempty"`
	RequestPolicy *CapacityRequestPolicy `json:"requestPolicy,omitempty"`
}

type CapacityRequestPolicy struct {
	Default     *resource.Quantity          `json:"default,omitempty"`
	ValidValues []resource.Quantity         `json:"validValues,omitempty"`
	ValidRange  *CapacityRequestPolicyRange `json:"validRange,omitempty"`
}

type CapacityRequestPolicyRange struct {
	Min  *resource.Quantity `json:"min,omitempty"`
	Max  *resource.Quantity `json:"max,omitempty"`
	Step *resource.Quantity `json:"step,omitempty"`
}

type Counter struct {
	Value resource.Quantity `json:"value"`
}
