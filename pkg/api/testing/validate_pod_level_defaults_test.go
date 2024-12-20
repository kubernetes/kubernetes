/*
Copyright 2025 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apisapps1 "k8s.io/kubernetes/pkg/apis/apps/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
)

func TestPodLevelResourcesDefaults(t *testing.T) {
	resCPU := v1.ResourceName(v1.ResourceCPU)
	valCPU1 := resource.MustParse("1")
	resMemory := v1.ResourceName(v1.ResourceMemory)
	valMem100Mi := resource.MustParse("100Mi")
	resHugepage2Mi := v1.ResourceName("hugepages-2Mi")
	valhugepage2Mi := resource.MustParse("2Mi")
	valhugepage10Mi := resource.MustParse("10Mi")
	resHugepage1Gi := v1.ResourceName("hugepages-1Gi")
	valHugepage1Gi := resource.MustParse("1Gi")

	testCases := []struct {
		name               string
		podResources       *v1.ResourceRequirements
		containerResources []v1.ResourceRequirements
		expectErr          bool
	}{
		{
			name: "no pod-level resources, container hugepages-2Mi:R",
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:    valCPU1,
						resMemory: valMem100Mi,
					},
					Requests: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
				},
			},
			expectErr: true,
		},
		{
			name: "no pod-level resources, container hugepages-2Mi:L",
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
					Requests: v1.ResourceList{
						resCPU:    valCPU1,
						resMemory: valMem100Mi,
					},
				},
			},
			expectErr: false,
		},
		{
			name: "no pod-level resources, container hugepages-2Mi:R/L",
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
					Requests: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
				},
			},
			expectErr: false,
		},
		{
			name: "no pod-level resources, container hugepages-2Mi:R/L hugepages-1Gi:R/L",
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
						resHugepage1Gi: valHugepage1Gi,
					},
					Requests: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
						resHugepage1Gi: valHugepage1Gi,
					},
				},
			},
			expectErr: false,
		},
		{
			name: "pod-level resources hugepages-2Mi:R, container hugepages-2Mi:none",
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					resCPU:    valCPU1,
					resMemory: valMem100Mi,
				},
				Requests: v1.ResourceList{
					resCPU:         valCPU1,
					resMemory:      valMem100Mi,
					resHugepage2Mi: valhugepage10Mi,
				},
			},
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:    valCPU1,
						resMemory: valMem100Mi,
					},
					Requests: v1.ResourceList{
						resCPU:    valCPU1,
						resMemory: valMem100Mi,
					},
				},
			},
			expectErr: true,
		},
		{
			name: "pod-level resources hugepages-2Mi:L, container hugepages-2Mi:L",
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					resCPU:         valCPU1,
					resMemory:      valMem100Mi,
					resHugepage2Mi: valhugepage10Mi,
				},
				Requests: v1.ResourceList{
					resCPU:    valCPU1,
					resMemory: valMem100Mi,
				},
			},
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
					Requests: v1.ResourceList{
						resCPU:    valCPU1,
						resMemory: valMem100Mi,
					},
				},
			},
			expectErr: false,
		},
		{
			name: "pod-level resources hugepages-2Mi:R/L, container hugepages-2Mi:R/L",
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					resCPU:         valCPU1,
					resMemory:      valMem100Mi,
					resHugepage2Mi: valhugepage10Mi,
				},
				Requests: v1.ResourceList{
					resCPU:         valCPU1,
					resMemory:      valMem100Mi,
					resHugepage2Mi: valhugepage10Mi,
				},
			},
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
					Requests: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
				},
			},
			expectErr: false,
		},
		{
			name: "pod-level resources hugepages-2Mi:R/L hugepages-1Gi:R/L, container hugepages-2Mi:R/L hugepages-1Gi:R/L",
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					resCPU:         valCPU1,
					resMemory:      valMem100Mi,
					resHugepage2Mi: valhugepage10Mi,
					resHugepage1Gi: valHugepage1Gi,
				},
				Requests: v1.ResourceList{
					resCPU:         valCPU1,
					resMemory:      valMem100Mi,
					resHugepage2Mi: valhugepage10Mi,
					resHugepage1Gi: valHugepage1Gi,
				},
			},
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
						resHugepage1Gi: valHugepage1Gi,
					},
					Requests: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
						resHugepage1Gi: valHugepage1Gi,
					},
				},
			},
			expectErr: false,
		},
		{
			name: "pod-level resources hugepages-2Mi:R, container hugepages-2Mi:L",
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					resCPU:    valCPU1,
					resMemory: valMem100Mi,
				},
				Requests: v1.ResourceList{
					resCPU:         valCPU1,
					resMemory:      valMem100Mi,
					resHugepage2Mi: valhugepage10Mi,
				},
			},
			containerResources: []v1.ResourceRequirements{
				{
					Limits: v1.ResourceList{
						resCPU:         valCPU1,
						resMemory:      valMem100Mi,
						resHugepage2Mi: valhugepage2Mi,
					},
					Requests: v1.ResourceList{
						resCPU:    valCPU1,
						resMemory: valMem100Mi,
					},
				},
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
			opts := validation.PodValidationOptions{PodLevelResourcesEnabled: true}

			// Step 1: Pod defaulting and validation
			podForPodValidation1 := makePod(tc.podResources, tc.containerResources)

			corev1.SetDefaults_Pod(podForPodValidation1)
			corev1.SetDefaults_PodSpec(&podForPodValidation1.Spec)

			internalPod := &core.Pod{}
			if err := legacyscheme.Scheme.Convert(podForPodValidation1, internalPod, nil); err != nil {
				t.Fatalf("Step 1: Failed to convert v1.Pod to core.Pod: %v", err)
			}

			podErrs := validation.ValidatePodSpec(&internalPod.Spec, &internalPod.ObjectMeta, field.NewPath("spec"), opts)
			if len(podErrs) > 0 != tc.expectErr {
				t.Errorf("Step 1: Pod validation failed. expectErr=%v, got errs: %v", tc.expectErr, podErrs.ToAggregate())
			}

			// Step 2: Deployment defaulting and validation
			podForPodValidation2 := makePod(tc.podResources, tc.containerResources)
			deployment := &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test-deploy", Namespace: "test-ns"},
				Spec: appsv1.DeploymentSpec{
					Template: v1.PodTemplateSpec{
						Spec: podForPodValidation2.Spec,
					},
				},
			}

			apisapps1.SetDefaults_Deployment(deployment)
			corev1.SetDefaults_PodSpec(&deployment.Spec.Template.Spec)
			internalPodTemplateSpec := &core.PodTemplateSpec{}
			if err := legacyscheme.Scheme.Convert(&deployment.Spec.Template, internalPodTemplateSpec, nil); err != nil {
				t.Fatalf("Step 2: Failed to convert v1.PodTemplateSpec to core.PodTemplateSpec: %v", err)
			}

			podErrs = validation.ValidatePodTemplateSpec(internalPodTemplateSpec, field.NewPath("template"), opts)
			if len(podErrs) > 0 != tc.expectErr {
				t.Errorf("Step 2: Pod Template validation failed. expectErr=%v, got errs: %v", tc.expectErr, podErrs.ToAggregate())
			}

			// Step 3: Validate the defaulted pod spec from the deployment
			podForPodValidation3 := &v1.Pod{
				Spec: deployment.Spec.Template.Spec,
			}
			corev1.SetDefaults_Pod(podForPodValidation3)
			corev1.SetDefaults_PodSpec(&podForPodValidation3.Spec)
			internalPod = &core.Pod{}
			if err := legacyscheme.Scheme.Convert(podForPodValidation3, internalPod, nil); err != nil {
				t.Fatalf("Step 3: Failed to convert v1.Pod to core.Pod: %v", err)
			}

			podErrs = validation.ValidatePodSpec(&internalPod.Spec, &internalPod.ObjectMeta, field.NewPath("spec"), opts)
			if len(podErrs) > 0 != tc.expectErr {
				t.Errorf("Step 3: Pod validation failed. expectErr=%v, got errs: %v", tc.expectErr, podErrs.ToAggregate())
			}
		})
	}
}

func makePod(podResources *v1.ResourceRequirements, containersResources []v1.ResourceRequirements) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "test-ns"},
		Spec: v1.PodSpec{
			Containers: makeContainers(containersResources),
		},
	}

	if podResources != nil {
		pod.Spec.Resources = podResources
	}

	return pod
}

func makeContainers(resourceRequirements []v1.ResourceRequirements) []v1.Container {
	containers := []v1.Container{}
	for idx, containerResources := range resourceRequirements {
		container := v1.Container{
			Name:  fmt.Sprintf("container-%d", idx),
			Image: "img", ImagePullPolicy: "Never", TerminationMessagePolicy: "File",
			Resources: containerResources,
		}
		containers = append(containers, container)
	}

	return containers
}
