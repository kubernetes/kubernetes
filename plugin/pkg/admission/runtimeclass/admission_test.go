/*
Copyright 2019 The Kubernetes Authors.

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

package runtimeclass

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/api/node/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

func validPod(name string, numContainers int, resources core.ResourceRequirements, setOverhead bool) *core.Pod {
	pod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test"},
		Spec:       core.PodSpec{},
	}
	pod.Spec.Containers = make([]core.Container, 0, numContainers)
	for i := 0; i < numContainers; i++ {
		pod.Spec.Containers = append(pod.Spec.Containers, core.Container{
			Image:     "foo:V" + strconv.Itoa(i),
			Resources: resources,
			Name:      "foo-" + strconv.Itoa(i),
		})
	}

	if setOverhead {
		pod.Spec.Overhead = core.ResourceList{
			core.ResourceName(core.ResourceCPU):    resource.MustParse("100m"),
			core.ResourceName(core.ResourceMemory): resource.MustParse("1"),
		}
	}
	return pod
}

func getGuaranteedRequirements() core.ResourceRequirements {
	resources := core.ResourceList{
		core.ResourceName(core.ResourceCPU):    resource.MustParse("1"),
		core.ResourceName(core.ResourceMemory): resource.MustParse("10"),
	}

	return core.ResourceRequirements{Limits: resources, Requests: resources}
}

func TestSetOverhead(t *testing.T) {

	tests := []struct {
		name         string
		runtimeClass *v1beta1.RuntimeClass
		pod          *core.Pod
		expectError  bool
		expectedPod  *core.Pod
	}{
		{
			name: "overhead, no container requirements",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("100m"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("1"),
					},
				},
			},
			pod:         validPod("no-resource-req-no-overhead", 1, core.ResourceRequirements{}, false),
			expectError: false,
			expectedPod: validPod("no-resource-req-no-overhead", 1, core.ResourceRequirements{}, true),
		},
		{
			name: "overhead, guaranteed pod",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("100m"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("1"),
					},
				},
			},
			pod:         validPod("guaranteed", 1, getGuaranteedRequirements(), false),
			expectError: false,
			expectedPod: validPod("guaranteed", 1, core.ResourceRequirements{}, true),
		},
		{
			name: "overhead, pod with differing overhead already set",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("10"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("10G"),
					},
				},
			},
			pod:         validPod("empty-requiremennts-overhead", 1, core.ResourceRequirements{}, true),
			expectError: true,
			expectedPod: nil,
		},
		{
			name: "overhead, pod with same overhead already set",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("100m"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("1"),
					},
				},
			},
			pod:         validPod("empty-requiremennts-overhead", 1, core.ResourceRequirements{}, true),
			expectError: false,
			expectedPod: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			attrs := admission.NewAttributesRecord(tc.pod, nil, core.Kind("Pod").WithVersion("version"), tc.pod.Namespace, tc.pod.Name, core.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{})

			errs := setOverhead(attrs, tc.pod, tc.runtimeClass)
			if tc.expectError {
				assert.NotEmpty(t, errs)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}
func NewObjectInterfacesForTest() admission.ObjectInterfaces {
	scheme := runtime.NewScheme()
	corev1.AddToScheme(scheme)
	return admission.NewObjectInterfacesFromScheme(scheme)
}

func TestValidate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodOverhead, true)()

	tests := []struct {
		name         string
		runtimeClass *v1beta1.RuntimeClass
		pod          *core.Pod
		expectError  bool
	}{
		{
			name: "No Overhead in RunntimeClass, Overhead set in pod",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
			},
			pod:         validPod("no-resource-req-no-overhead", 1, getGuaranteedRequirements(), true),
			expectError: true,
		},
		{
			name: "Non-matching Overheads",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("10"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("10G"),
					},
				},
			},
			pod:         validPod("no-resource-req-no-overhead", 1, core.ResourceRequirements{}, true),
			expectError: true,
		},
		{
			name: "Matching Overheads",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("100m"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("1"),
					},
				},
			},
			pod:         validPod("no-resource-req-no-overhead", 1, core.ResourceRequirements{}, false),
			expectError: false,
		},
	}
	rt := NewRuntimeClass()
	o := NewObjectInterfacesForTest()
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			attrs := admission.NewAttributesRecord(tc.pod, nil, core.Kind("Pod").WithVersion("version"), tc.pod.Namespace, tc.pod.Name, core.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{})

			errs := rt.Validate(attrs, o)
			if tc.expectError {
				assert.NotEmpty(t, errs)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestValidateOverhead(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodOverhead, true)()

	tests := []struct {
		name         string
		runtimeClass *v1beta1.RuntimeClass
		pod          *core.Pod
		expectError  bool
	}{
		{
			name: "Overhead part of RuntimeClass, no Overhead defined in pod",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("100m"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("1"),
					},
				},
			},
			pod:         validPod("no-requirements", 1, core.ResourceRequirements{}, false),
			expectError: true,
		},
		{
			name: "No Overhead in RunntimeClass, Overhead set in pod",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
			},
			pod:         validPod("no-resource-req-no-overhead", 1, getGuaranteedRequirements(), true),
			expectError: true,
		},
		{
			name:         "No RunntimeClass, Overhead set in pod",
			runtimeClass: nil,
			pod:          validPod("no-resource-req-no-overhead", 1, getGuaranteedRequirements(), true),
			expectError:  true,
		},
		{
			name: "Non-matching Overheads",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("10"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("10G"),
					},
				},
			},
			pod:         validPod("no-resource-req-no-overhead", 1, core.ResourceRequirements{}, true),
			expectError: true,
		},
		{
			name: "Matching Overheads",
			runtimeClass: &v1beta1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Overhead: &v1beta1.Overhead{
					PodFixed: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("100m"),
						corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("1"),
					},
				},
			},
			pod:         validPod("no-resource-req-no-overhead", 1, core.ResourceRequirements{}, true),
			expectError: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			attrs := admission.NewAttributesRecord(tc.pod, nil, core.Kind("Pod").WithVersion("version"), tc.pod.Namespace, tc.pod.Name, core.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, &user.DefaultInfo{})

			errs := validateOverhead(attrs, tc.pod, tc.runtimeClass)
			if tc.expectError {
				assert.NotEmpty(t, errs)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}
