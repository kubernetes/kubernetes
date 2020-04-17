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

package core

import (
	"k8s.io/apimachinery/pkg/api/resource"
)

func (rn ResourceName) String() string {
	return string(rn)
}

// CPU returns the CPU limit if specified.
func (rl *ResourceList) CPU() *resource.Quantity {
	if val, ok := (*rl)[ResourceCPU]; ok {
		return &val
	}
	return &resource.Quantity{Format: resource.DecimalSI}
}

// Memory returns the Memory limit if specified.
func (rl *ResourceList) Memory() *resource.Quantity {
	if val, ok := (*rl)[ResourceMemory]; ok {
		return &val
	}
	return &resource.Quantity{Format: resource.BinarySI}
}

// Storage returns the Storage limit if specified.
func (rl *ResourceList) Storage() *resource.Quantity {
	if val, ok := (*rl)[ResourceStorage]; ok {
		return &val
	}
	return &resource.Quantity{Format: resource.BinarySI}
}

// Pods returns the list of pods
func (rl *ResourceList) Pods() *resource.Quantity {
	if val, ok := (*rl)[ResourcePods]; ok {
		return &val
	}
	return &resource.Quantity{}
}

// StorageEphemeral returns the list of ephemeral storage volumes, if any
func (rl *ResourceList) StorageEphemeral() *resource.Quantity {
	if val, ok := (*rl)[ResourceEphemeralStorage]; ok {
		return &val
	}
	return &resource.Quantity{}
}
