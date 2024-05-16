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
	return rl.Name(ResourceCPU, resource.DecimalSI)
}

// Memory returns the Memory limit if specified.
func (rl *ResourceList) Memory() *resource.Quantity {
	return rl.Name(ResourceMemory, resource.BinarySI)
}

// Storage returns the Storage limit if specified.
func (rl *ResourceList) Storage() *resource.Quantity {
	return rl.Name(ResourceStorage, resource.BinarySI)
}

// Pods returns the list of pods
func (rl *ResourceList) Pods() *resource.Quantity {
	return rl.Name(ResourcePods, resource.DecimalSI)
}

// StorageEphemeral returns the list of ephemeral storage volumes, if any
func (rl *ResourceList) StorageEphemeral() *resource.Quantity {
	return rl.Name(ResourceEphemeralStorage, resource.BinarySI)
}

// Name returns the resource with name if specified, otherwise it returns a nil quantity with default format.
func (rl *ResourceList) Name(name ResourceName, defaultFormat resource.Format) *resource.Quantity {
	if val, ok := (*rl)[name]; ok {
		return &val
	}
	return &resource.Quantity{Format: defaultFormat}
}
