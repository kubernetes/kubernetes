/*
Copyright 2014 Google Inc. All rights reserved.

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

package resourcedefaults

import (
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func init() {
	admission.RegisterPlugin("ResourceDefaults", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewResourceDefaults(), nil
	})
}

const (
	defaultMemory string = "512Mi"
	defaultCPU    string = "1"
)

// resourceDefaults is an implementation of admission.Interface which applies default resource limits (cpu/memory)
// It is useful for clusters that do not want to support unlimited usage constraints, but instead supply sensible defaults
type resourceDefaults struct{}

func (resourceDefaults) Admit(a admission.Attributes) (err error) {
	// ignore deletes, only process create and update
	if a.GetOperation() == "DELETE" {
		return nil
	}

	// we only care about pods
	if a.GetResource() != "pods" {
		return nil
	}

	// get the pod, so we can validate each of the containers within have default mem / cpu constraints
	obj := a.GetObject()
	pod := obj.(*api.Pod)
	for index := range pod.Spec.Containers {
		if pod.Spec.Containers[index].Resources.Limits == nil {
			pod.Spec.Containers[index].Resources.Limits = api.ResourceList{}
		}
		if pod.Spec.Containers[index].Resources.Limits.Memory().Value() == 0 {
			pod.Spec.Containers[index].Resources.Limits[api.ResourceMemory] = resource.MustParse(defaultMemory)
		}
		if pod.Spec.Containers[index].Resources.Limits.Cpu().Value() == 0 {
			pod.Spec.Containers[index].Resources.Limits[api.ResourceCPU] = resource.MustParse(defaultCPU)
		}
	}
	return nil
}

func NewResourceDefaults() admission.Interface {
	return new(resourceDefaults)
}
