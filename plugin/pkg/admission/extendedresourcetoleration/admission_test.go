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

package extendedresourcetoleration

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

func TestAdmit(t *testing.T) {

	plugin := newExtendedResourceToleration()

	containerRequestingCPU := core.Container{
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceCPU: *resource.NewQuantity(2, resource.DecimalSI),
			},
		},
	}

	containerRequestingMemory := core.Container{
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceMemory: *resource.NewQuantity(2048, resource.DecimalSI),
			},
		},
	}

	extendedResource1 := "example.com/device-ek"
	extendedResource2 := "example.com/device-do"

	containerRequestingExtendedResource1 := core.Container{
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(extendedResource1): *resource.NewQuantity(1, resource.DecimalSI),
			},
		},
	}
	containerRequestingExtendedResource2 := core.Container{
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(extendedResource2): *resource.NewQuantity(2, resource.DecimalSI),
			},
		},
	}

	tests := []struct {
		description  string
		requestedPod core.Pod
		expectedPod  core.Pod
	}{
		{
			description: "empty pod without any extended resources, expect no change in tolerations",
			requestedPod: core.Pod{
				Spec: core.PodSpec{},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{},
			},
		},
		{
			description: "pod with container without any extended resources, expect no change in tolerations",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
					},
				},
			},
		},
		{
			description: "pod with init container without any extended resources, expect no change in tolerations",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						containerRequestingMemory,
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						containerRequestingMemory,
					},
				},
			},
		},
		{
			description: "pod with container with extended resource, expect toleration to be added",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingExtendedResource1,
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			description: "pod with init container with extended resource, expect toleration to be added",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						containerRequestingExtendedResource2,
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						containerRequestingExtendedResource2,
					},
					Tolerations: []core.Toleration{
						{
							Key:      extendedResource2,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			description: "pod with existing tolerations and container with extended resource, expect existing tolerations to be preserved and new toleration to be added",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Key:      "foo",
							Operator: core.TolerationOpEqual,
							Value:    "bar",
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Key:      "foo",
							Operator: core.TolerationOpEqual,
							Value:    "bar",
							Effect:   core.TaintEffectNoSchedule,
						},
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			description: "pod with multiple extended resources, expect multiple tolerations to be added",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					InitContainers: []core.Container{
						containerRequestingCPU,
						containerRequestingExtendedResource2,
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					InitContainers: []core.Container{
						containerRequestingCPU,
						containerRequestingExtendedResource2,
					},
					Tolerations: []core.Toleration{
						// Note the order, it's sorted by the Key
						{
							Key:      extendedResource2,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			description: "pod with container requesting extended resource and existing correct toleration, expect no change in tolerations",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			description: "pod with container requesting extended resource and existing toleration with the same key but different effect and value, expect existing tolerations to be preserved and new toleration to be added",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpEqual,
							Value:    "foo",
							Effect:   core.TaintEffectNoExecute,
						},
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpEqual,
							Value:    "foo",
							Effect:   core.TaintEffectNoExecute,
						},
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			description: "pod with wildcard toleration and container requesting extended resource, expect existing tolerations to be preserved and new toleration to be added",
			requestedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Operator: core.TolerationOpExists,
						},
					},
				},
			},
			expectedPod: core.Pod{
				Spec: core.PodSpec{
					Containers: []core.Container{
						containerRequestingCPU,
						containerRequestingMemory,
						containerRequestingExtendedResource1,
					},
					Tolerations: []core.Toleration{
						{
							Operator: core.TolerationOpExists,
						},
						{
							Key:      extendedResource1,
							Operator: core.TolerationOpExists,
							Effect:   core.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
	}
	for i, test := range tests {
		err := plugin.Admit(admission.NewAttributesRecord(&test.requestedPod, nil, core.Kind("Pod").WithVersion("version"), "foo", "name", core.Resource("pods").WithVersion("version"), "", "ignored", false, nil))
		if err != nil {
			t.Errorf("[%d: %s] unexpected error %v for pod %+v", i, test.description, err, test.requestedPod)
		}

		if !helper.Semantic.DeepEqual(test.expectedPod.Spec.Tolerations, test.requestedPod.Spec.Tolerations) {
			t.Errorf("[%d: %s] expected %#v got %#v", i, test.description, test.expectedPod.Spec.Tolerations, test.requestedPod.Spec.Tolerations)
		}
	}
}

func TestHandles(t *testing.T) {
	plugin := newExtendedResourceToleration()
	tests := map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  true,
		admission.Delete:  false,
		admission.Connect: false,
	}
	for op, expected := range tests {
		result := plugin.Handles(op)
		if result != expected {
			t.Errorf("Unexpected result for operation %s: %v\n", op, result)
		}
	}
}
