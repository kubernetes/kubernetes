/*
Copyright 2020 The Kubernetes Authors.

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

package podresourceallocation

import (
	"context"
	"fmt"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestAdmitCreate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()
	namespace := "test"
	handler := NewPodResourceAllocation()
	pod := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: namespace},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "c1",
					Image: "image",
				},
			},
		},
	}
	res := api.ResourceList{
		api.ResourceCPU:    resource.MustParse("1"),
		api.ResourceMemory: resource.MustParse("1Gi"),
	}
	cpuPolicyNoRestart := api.ResizePolicy{ResourceName: api.ResourceCPU, Policy: api.NoRestart}
	memPolicyNoRestart := api.ResizePolicy{ResourceName: api.ResourceMemory, Policy: api.NoRestart}
	cpuPolicyRestart := api.ResizePolicy{ResourceName: api.ResourceCPU, Policy: api.RestartContainer}
	memPolicyRestart := api.ResizePolicy{ResourceName: api.ResourceMemory, Policy: api.RestartContainer}
	tests := []struct {
		name                       string
		resources                  api.ResourceRequirements
		resourcesAllocated         api.ResourceList
		expectedResourcesAllocated api.ResourceList
		resizePolicy               []api.ResizePolicy
		expectedResizePolicy       []api.ResizePolicy
	}{
		{
			name:                       "create new pod - resource allocation not set, resize policy not set",
			resources:                  api.ResourceRequirements{Requests: res, Limits: res},
			resourcesAllocated:         nil,
			expectedResourcesAllocated: res,
			resizePolicy:               []api.ResizePolicy{},
			expectedResizePolicy:       []api.ResizePolicy{cpuPolicyNoRestart, memPolicyNoRestart},
		},
		{
			name:                       "create new pod - resource allocation equals desired, norestart resize policy set",
			resources:                  api.ResourceRequirements{Requests: res, Limits: res},
			resourcesAllocated:         res,
			expectedResourcesAllocated: res,
			resizePolicy:               []api.ResizePolicy{cpuPolicyNoRestart, memPolicyNoRestart},
			expectedResizePolicy:       []api.ResizePolicy{cpuPolicyNoRestart, memPolicyNoRestart},
		},
		{
			name:                       "create new pod - resources & resource allocation not set, cpu restart resize policy set",
			resources:                  api.ResourceRequirements{},
			resourcesAllocated:         nil,
			expectedResourcesAllocated: nil,
			resizePolicy:               []api.ResizePolicy{cpuPolicyRestart},
			expectedResizePolicy:       []api.ResizePolicy{},
		},
		{
			name:                       "create new pod - resource allocation equals requests, mem restart resize policy set",
			resources:                  api.ResourceRequirements{Requests: res},
			resourcesAllocated:         res,
			expectedResourcesAllocated: res,
			resizePolicy:               []api.ResizePolicy{memPolicyRestart},
			expectedResizePolicy:       []api.ResizePolicy{memPolicyRestart, cpuPolicyNoRestart},
		},
		{
			name:                       "create new pod - resource allocation not set, cpu & mem restart resize policy set",
			resources:                  api.ResourceRequirements{Requests: res},
			resourcesAllocated:         nil,
			expectedResourcesAllocated: res,
			resizePolicy:               []api.ResizePolicy{cpuPolicyRestart, memPolicyRestart},
			expectedResizePolicy:       []api.ResizePolicy{cpuPolicyRestart, memPolicyRestart},
		},
		//TODO: look into if more unit tests and negative tests could be added
	}

	for _, tc := range tests {
		pod.Spec.Containers[0].Resources = tc.resources
		pod.Spec.Containers[0].ResourcesAllocated = tc.resourcesAllocated
		pod.Spec.Containers[0].ResizePolicy = tc.resizePolicy
		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"),
			pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "",
			admission.Create, nil, false, nil), nil)
		if !apiequality.Semantic.DeepEqual(pod.Spec.Containers[0].ResourcesAllocated, tc.expectedResourcesAllocated) {
			t.Fatal(fmt.Sprintf("Test: %s - resourcesAllocated mismatch\nExpected: %+v\nGot: %+v\nError: %+v", tc.name,
				tc.expectedResourcesAllocated, pod.Spec.Containers[0].ResourcesAllocated, err))
		}
		if !apiequality.Semantic.DeepEqual(pod.Spec.Containers[0].ResizePolicy, tc.expectedResizePolicy) {
			t.Fatal(fmt.Sprintf("Test: %s - resizePolicy mismatch\nExpected: %+v\nGot: %+v\nError: %+v", tc.name,
				tc.expectedResizePolicy, pod.Spec.Containers[0].ResizePolicy, err))
		}
	}
}

func TestAdmitUpdate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()
	namespace := "test"
	handler := NewPodResourceAllocation()
	res := api.ResourceList{
		api.ResourceCPU:    resource.MustParse("2"),
		api.ResourceMemory: resource.MustParse("2Gi"),
	}
	res2 := api.ResourceList{
		api.ResourceCPU:    resource.MustParse("1"),
		api.ResourceMemory: resource.MustParse("1Gi"),
	}
	oldPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: namespace},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:      "c1",
					Image:     "image",
					Resources: api.ResourceRequirements{Requests: res, Limits: res},
				},
			},
		},
	}
	cpuPolicyNoRestart := api.ResizePolicy{ResourceName: api.ResourceCPU, Policy: api.NoRestart}
	memPolicyNoRestart := api.ResizePolicy{ResourceName: api.ResourceMemory, Policy: api.NoRestart}
	cpuPolicyRestart := api.ResizePolicy{ResourceName: api.ResourceCPU, Policy: api.RestartContainer}
	memPolicyRestart := api.ResizePolicy{ResourceName: api.ResourceMemory, Policy: api.RestartContainer}
	tests := []struct {
		name                       string
		oldResourcesAllocated      api.ResourceList
		newResourcesAllocated      api.ResourceList
		expectedResourcesAllocated api.ResourceList
		oldResizePolicy            []api.ResizePolicy
		newResizePolicy            []api.ResizePolicy
		expectedResizePolicy       []api.ResizePolicy
	}{
		{
			name:                       "update pod - resource allocation dropped, resize policy dropped (nil)",
			oldResourcesAllocated:      res,
			newResourcesAllocated:      nil,
			expectedResourcesAllocated: res,
			oldResizePolicy:            []api.ResizePolicy{cpuPolicyNoRestart, memPolicyNoRestart},
			newResizePolicy:            nil,
			expectedResizePolicy:       []api.ResizePolicy{cpuPolicyNoRestart, memPolicyNoRestart},
		},
		{
			name:                       "update pod - resource allocation not set, resize policy dropped (empty)",
			oldResourcesAllocated:      nil,
			newResourcesAllocated:      nil,
			expectedResourcesAllocated: nil,
			oldResizePolicy:            []api.ResizePolicy{cpuPolicyRestart, memPolicyRestart},
			newResizePolicy:            []api.ResizePolicy{},
			expectedResizePolicy:       []api.ResizePolicy{cpuPolicyRestart, memPolicyRestart},
		},
		{
			name:                       "update pod - resource allocation retained, resize policy dropped (nil)",
			oldResourcesAllocated:      res,
			newResourcesAllocated:      res,
			expectedResourcesAllocated: res,
			oldResizePolicy:            []api.ResizePolicy{cpuPolicyNoRestart, memPolicyRestart},
			newResizePolicy:            nil,
			expectedResizePolicy:       []api.ResizePolicy{cpuPolicyNoRestart, memPolicyRestart},
		},
		{
			name:                       "update pod - resource allocation not equal to desired dropped, resize policy dropped (empty)",
			oldResourcesAllocated:      res2,
			newResourcesAllocated:      nil,
			expectedResourcesAllocated: res2,
			oldResizePolicy:            []api.ResizePolicy{cpuPolicyRestart, memPolicyNoRestart},
			newResizePolicy:            []api.ResizePolicy{},
			expectedResizePolicy:       []api.ResizePolicy{cpuPolicyRestart, memPolicyNoRestart},
		},
		//TODO: look into if more unit tests can be added
	}

	for _, tc := range tests {
		newPod := oldPod.DeepCopy()
		oldPod.Spec.Containers[0].ResourcesAllocated = tc.oldResourcesAllocated
		oldPod.Spec.Containers[0].ResizePolicy = tc.oldResizePolicy
		newPod.Spec.Containers[0].ResourcesAllocated = tc.newResourcesAllocated
		newPod.Spec.Containers[0].ResizePolicy = tc.newResizePolicy
		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(newPod, oldPod, api.Kind("Pod").WithVersion("version"),
			newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "",
			admission.Update, nil, false, nil), nil)
		if !apiequality.Semantic.DeepEqual(newPod.Spec.Containers[0].ResourcesAllocated, tc.expectedResourcesAllocated) {
			t.Fatal(fmt.Sprintf("Test: %s - resourcesAllocated mismatch\nExpected: %+v\nGot: %+v\nError: %+v", tc.name,
				tc.expectedResourcesAllocated, newPod.Spec.Containers[0].ResourcesAllocated, err))
		}
		if !apiequality.Semantic.DeepEqual(newPod.Spec.Containers[0].ResizePolicy, tc.expectedResizePolicy) {
			t.Fatal(fmt.Sprintf("Test: %s - resizePolicy mismatch\nExpected: %+v\nGot: %+v\nError: %+v", tc.name,
				tc.expectedResizePolicy, newPod.Spec.Containers[0].ResizePolicy, err))
		}
	}
}

func TestValidateCreate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()
	namespace := "test"
	handler := NewPodResourceAllocation()
	pod := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: namespace},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "c1",
					Image: "image",
				},
			},
		},
	}
	resources1 := api.ResourceList{
		api.ResourceCPU:    resource.MustParse("1"),
		api.ResourceMemory: resource.MustParse("1Gi"),
	}
	resources2 := api.ResourceList{
		api.ResourceCPU:    resource.MustParse("2"),
		api.ResourceMemory: resource.MustParse("2Gi"),
	}
	tests := []struct {
		name               string
		resources          api.ResourceRequirements
		resourcesAllocated api.ResourceList
		expectError        bool
	}{
		{
			name:               "create new pod - resource allocation not set",
			resources:          api.ResourceRequirements{Requests: resources1, Limits: resources1},
			resourcesAllocated: nil,
			expectError:        true,
		},
		{
			name:               "create new pod - resource allocation equals desired resources",
			resources:          api.ResourceRequirements{Requests: resources1, Limits: resources1},
			resourcesAllocated: resources1,
			expectError:        false,
		},
		{
			name:               "create new pod - resource allocation exceeds desired resources",
			resources:          api.ResourceRequirements{Requests: resources1, Limits: resources1},
			resourcesAllocated: resources2,
			expectError:        true,
		},
		//TODO: more unit tests and negative tests
	}

	for _, tc := range tests {
		pod.Spec.Containers[0].Resources = tc.resources
		pod.Spec.Containers[0].ResourcesAllocated = tc.resourcesAllocated
		err := handler.Validate(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"),
			pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "",
			admission.Create, nil, false, nil), nil)
		if tc.expectError && err == nil {
			t.Fatal(fmt.Sprintf("Test: %s - missing expected error", tc.name))
		}
		if !tc.expectError && err != nil {
			t.Fatal(fmt.Sprintf("Test: %s - received unexpected error %+v", tc.name, err))
		}
	}
}

func TestValidateUpdate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()
	namespace := "test"
	handler := NewPodResourceAllocation()
	resources := api.ResourceList{
		api.ResourceCPU:    resource.MustParse("1"),
		api.ResourceMemory: resource.MustParse("1Gi"),
	}
	oldPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: namespace},
		Spec: api.PodSpec{
			NodeName: "foonode",
			Containers: []api.Container{
				{
					Name:      "c1",
					Image:     "image",
					Resources: api.ResourceRequirements{Requests: resources, Limits: resources},
				},
			},
		},
	}
	newPod := oldPod.DeepCopy()
	newPod.Spec.Containers[0].ResourcesAllocated = resources
	tests := []struct {
		name        string
		userInfo    user.Info
		expectError bool
	}{
		{
			name:        "update existing pod - system:node user",
			userInfo:    &user.DefaultInfo{Name: "system:node:foonode", Groups: []string{user.AllAuthenticated, user.NodesGroup}},
			expectError: false,
		},
		{
			name:        "update existing pod not owned by node - system:node user",
			userInfo:    &user.DefaultInfo{Name: "system:node:barnode", Groups: []string{user.AllAuthenticated, user.NodesGroup}},
			expectError: true,
		},
		{
			name:        "update existing pod - system:admin user",
			userInfo:    &user.DefaultInfo{Name: "system:admin", Groups: []string{user.AllAuthenticated, user.SystemPrivilegedGroup}},
			expectError: true,
		},
		//TODO: more unit tests and negative tests
	}

	for _, tc := range tests {
		err := handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, oldPod, api.Kind("Pod").WithVersion("version"),
			newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "",
			admission.Update, nil, false, tc.userInfo), nil)
		if tc.expectError && err == nil {
			t.Fatal(fmt.Sprintf("Test: %s - missing expected error", tc.name))
		}
		if !tc.expectError && err != nil {
			t.Fatal(fmt.Sprintf("Test: %s - received unexpected error %+v", tc.name, err))
		}
	}
}

func TestHandles(t *testing.T) {
	handler := NewPodResourceAllocation()
	//TODO: Connect, Delete operation negative tests
	tests := []admission.Operation{admission.Create, admission.Update}

	for _, test := range tests {
		if !handler.Handles(test) {
			t.Errorf("Expected handling all operations, including: %v", test)
		}
	}
}
