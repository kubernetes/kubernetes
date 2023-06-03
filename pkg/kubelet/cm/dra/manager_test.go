/*
Copyright 2023 The Kubernetes Authors.

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

package dra

import (
	"context"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"testing"
)

func TestNewManagerImpl(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	stateFileDirectory := t.TempDir()
	manager, err := NewManagerImpl(kubeClient, stateFileDirectory)
	if err != nil {
		t.Errorf("Failed to create DRA manager: %v", err)
	}

	if manager.cache == nil {
		t.Error("Expected cache to be initialized, but it was nil")
	}

	if manager.kubeClient != kubeClient {
		t.Error("Expected kubeClient to be set, but it was not")
	}
}

func TestGetResources(t *testing.T) {
	rscName := "test-pod-claim-1"
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
		Spec: v1.PodSpec{
			ResourceClaims: []v1.PodResourceClaim{
				{
					Name:   "test-pod-claim-1",
					Source: v1.ClaimSource{ResourceClaimName: &rscName},
				},
			},
		},
	}

	container := &v1.Container{
		Name: "test-container",
		Resources: v1.ResourceRequirements{
			Claims: []v1.ResourceClaim{
				{
					Name: "test-pod-claim-1",
				},
			},
		},
	}

	kubeClient := fake.NewSimpleClientset()
	stateFileDirectory := t.TempDir()
	manager, err := NewManagerImpl(kubeClient, stateFileDirectory)
	if err != nil {
		t.Errorf("Failed to create DRA manager: %v", err)
	}

	var annotation []kubecontainer.Annotation
	annotation = []kubecontainer.Annotation{
		{
			Name:  "test-annotation",
			Value: "123",
		},
	}

	var cis state.ClaimInfoState
	cis = state.ClaimInfoState{
		ClaimName: "test-pod-claim-1",
		CDIDevices: map[string][]string{
			"test-cdi-device": {"123"},
		},
		Namespace: "test-namespace",
	}

	manager.cache.add(&ClaimInfo{annotations: annotation, ClaimInfoState: cis})
	manager.cache.add(&ClaimInfo{annotations: annotation})

	_, err = manager.GetResources(pod, container)
	if err != nil {
		t.Fatalf("error occur: %v", err)
	}
}

func TestPrepareResources(t *testing.T) {
	// Create a fake kubeClient for testing
	fakeKubeClient := fake.NewSimpleClientset()

	cache, err := newClaimInfoCache(t.TempDir(), draManagerStateFileName)
	if err != nil {
		t.Fatalf("failed to newClaimInfoCache, err:%v", err)
	}

	// Create a ManagerImpl instance with the fake kubeClient
	manager := &ManagerImpl{
		kubeClient: fakeKubeClient,
		cache:      cache,
	}

	rscName := "test-resource-claim"
	// Create a Pod with a ResourceClaimSpec
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
			UID:       "test-reserved",
		},
		Spec: v1.PodSpec{
			ResourceClaims: []v1.PodResourceClaim{
				{
					Name:   "test-pod-claim-1",
					Source: v1.ClaimSource{ResourceClaimName: &rscName},
				},
			},
		},
	}

	// Create a ResourceClaim with a DriverName and a ResourceHandle
	resourceClaim := &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-resource-claim",
			Namespace: "test-namespace",
			UID:       "test-reserved",
		},
		Spec: resourcev1alpha2.ResourceClaimSpec{
			ResourceClassName: "test-class",
		},
		Status: resourcev1alpha2.ResourceClaimStatus{
			DriverName: "test-driver",
			Allocation: &resourcev1alpha2.AllocationResult{
				ResourceHandles: []resourcev1alpha2.ResourceHandle{
					{Data: "test-data", DriverName: "test-driver"},
				},
			},
			ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
				{UID: "test-reserved"},
			},
		},
	}

	_, err = fakeKubeClient.ResourceV1alpha2().ResourceClaims(pod.Namespace).Create(context.Background(), resourceClaim, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create ResourceClaim: %+v", err)
	}

	plg := plugin.NewRegistrationHandler()
	err = plg.RegisterPlugin("test-driver", "127.0.0.1", []string{"1.27"}) //this place won't pass the CI
	if err != nil {
		t.Fatalf("failed to register plugin err:%v", err)
	}

	// Call the PrepareResources method
	err = manager.PrepareResources(pod)
	if err != nil {
		t.Fatalf("error:%v", err)
	}

	// check the cache contains the expected claim info
	claimName := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[0])
	claimInfo := manager.cache.get(claimName, pod.Namespace)
	if claimInfo == nil {
		t.Fatalf("claimInfo not found in cache for claim %s", claimName)
	}
	if claimInfo.DriverName != resourceClaim.Status.DriverName {
		t.Fatalf("driverName mismatch: expected %s, got %s", resourceClaim.Status.DriverName, claimInfo.DriverName)
	}
	if claimInfo.ClassName != resourceClaim.Spec.ResourceClassName {
		t.Fatalf("resourceClassName mismatch: expected %s, got %s", resourceClaim.Spec.ResourceClassName, claimInfo.ClassName)
	}
	if len(claimInfo.PodUIDs) != 1 || !claimInfo.PodUIDs.Has(string(pod.UID)) {
		t.Fatalf("podUIDs mismatch: expected [%s], got %v", pod.UID, claimInfo.PodUIDs)
	}
	if len(claimInfo.CDIDevices[resourceClaim.Status.DriverName]) != 1 || claimInfo.CDIDevices[resourceClaim.Status.DriverName][0] != "test-cdi-device" {
		t.Fatalf("cdiDevices mismatch: expected [%s], got %v", []string{"test-cdi-device"}, claimInfo.CDIDevices[resourceClaim.Status.DriverName])
	}
}

func TestUnprepareResources(t *testing.T) {
	// Create a fake kubeClient for testing
	fakeKubeClient := fake.NewSimpleClientset()

	cache, err := newClaimInfoCache(t.TempDir(), draManagerStateFileName)
	if err != nil {
		t.Fatalf("error occur:%v", err)
	}

	// Create a ManagerImpl instance with the fake kubeClient
	manager := &ManagerImpl{
		kubeClient: fakeKubeClient,
		cache:      cache,
	}

	rscName := "test-resource-claim"
	// Create a Pod with a ResourceClaimSpec
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
		Spec: v1.PodSpec{
			ResourceClaims: []v1.PodResourceClaim{
				{
					Name:   "test-pod-claim-1",
					Source: v1.ClaimSource{ResourceClaimName: &rscName},
				},
			},
		},
	}

	// Create a ResourceClaim with a DriverName and a ResourceHandle
	resourceClaim := &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-resource-claim",
			Namespace: "test-namespace",
			UID:       "test-reserved",
		},
		Spec: resourcev1alpha2.ResourceClaimSpec{
			ResourceClassName: "test-class",
		},
		Status: resourcev1alpha2.ResourceClaimStatus{
			DriverName: "test-driver",
			Allocation: &resourcev1alpha2.AllocationResult{
				ResourceHandles: []resourcev1alpha2.ResourceHandle{
					{Data: "test-data", DriverName: "test-driver"},
				},
			},
			ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
				{UID: "test-reserved"},
			},
		},
	}

	_, err = fakeKubeClient.ResourceV1alpha2().ResourceClaims(pod.Namespace).Create(context.Background(), resourceClaim, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create ResourceClaim: %+v", err)
	}

	manager.cache.add(&ClaimInfo{ClaimInfoState: state.ClaimInfoState{ClaimName: "test-resource-claim", Namespace: "test-namespace"}})

	// Call the UnprepareResources method
	err = manager.UnprepareResources(pod)
	if err != nil {
		return
	}

	// Check that the cache has been updated correctly
	claimName := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[0])
	claimInfo := manager.cache.get(claimName, pod.Namespace)
	if claimInfo != nil {
		t.Fatalf("claimInfo still found in cache after calling UnprepareResources")
	}

}

func TestGetContainerClaimInfos(t *testing.T) {

	cache, err := newClaimInfoCache(t.TempDir(), draManagerStateFileName)
	if err != nil {
		t.Fatalf("error occur:%v", err)
	}
	manager := &ManagerImpl{
		cache: cache,
	}

	rscName := "test-resource-claim-1"
	rscName2 := "test-resource-claim-2"
	// Create a test pod and container with resource claims
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			ResourceClaims: []v1.PodResourceClaim{
				{
					Name:   "claim1",
					Source: v1.ClaimSource{ResourceClaimName: &rscName},
				},
				{
					Name:   "claim2",
					Source: v1.ClaimSource{ResourceClaimName: &rscName2},
				},
			},
		},
	}
	container := &v1.Container{
		Resources: v1.ResourceRequirements{
			Claims: []v1.ResourceClaim{
				{
					Name: "claim1",
				},
				{
					Name: "claim2",
				},
			},
		},
	}

	// Add test fakeClaimInfos to the cache
	manager.cache.add(&ClaimInfo{ClaimInfoState: state.ClaimInfoState{ClaimName: "test-resource-claim-1"}})
	manager.cache.add(&ClaimInfo{ClaimInfoState: state.ClaimInfoState{ClaimName: "test-resource-claim-2"}})

	// Call the GetContainerClaimInfos() function
	fakeClaimInfos, err := manager.GetContainerClaimInfos(pod, container)

	// Check the results
	assert.NoError(t, err)
	assert.Equal(t, 2, len(fakeClaimInfos))
	assert.Equal(t, "test-resource-claim-1", fakeClaimInfos[0].ClaimInfoState.ClaimName)
	assert.Equal(t, "test-resource-claim-2", fakeClaimInfos[1].ClaimInfoState.ClaimName)

	// Test with a missing claimInfo
	manager.cache.delete("claim2", "default")
	_, err = manager.GetContainerClaimInfos(pod, container)
	assert.NoError(t, err)
}
