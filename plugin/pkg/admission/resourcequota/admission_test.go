/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package resourcequota

import (
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller/resourcequota"
)

func getResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func getResourceRequirements(requests, limits api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func validPod(name string, numContainers int, resources api.ResourceRequirements) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "test"},
		Spec:       api.PodSpec{},
	}
	pod.Spec.Containers = make([]api.Container, 0, numContainers)
	for i := 0; i < numContainers; i++ {
		pod.Spec.Containers = append(pod.Spec.Containers, api.Container{
			Image:     "foo:V" + strconv.Itoa(i),
			Resources: resources,
		})
	}
	return pod
}

func TestAdmissionIgnoresDelete(t *testing.T) {
	namespace := "default"
	handler := createResourceQuota(&testclient.Fake{}, nil)
	err := handler.Admit(admission.NewAttributesRecord(nil, "Pod", namespace, "name", "pods", "", admission.Delete, nil))
	if err != nil {
		t.Errorf("ResourceQuota should admit all deletes: %v", err)
	}
}

func TestAdmissionIgnoresSubresources(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	handler := createResourceQuota(&testclient.Fake{}, indexer)

	quota := &api.ResourceQuota{}
	quota.Name = "quota"
	quota.Namespace = "test"
	quota.Status = api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	quota.Status.Hard[api.ResourceMemory] = resource.MustParse("2Gi")
	quota.Status.Used[api.ResourceMemory] = resource.MustParse("1Gi")

	indexer.Add(quota)

	newPod := validPod("123", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, "Pod", newPod.Namespace, newPod.Name, "pods", "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected an error because the pod exceeded allowed quota")
	}

	err = handler.Admit(admission.NewAttributesRecord(newPod, "Pod", newPod.Namespace, newPod.Name, "pods", "subresource", admission.Create, nil))
	if err != nil {
		t.Errorf("Did not expect an error because the action went to a subresource: %v", err)
	}

}

func TestIncrementUsagePodResources(t *testing.T) {
	type testCase struct {
		testName      string
		existing      *api.Pod
		input         *api.Pod
		resourceName  api.ResourceName
		hard          resource.Quantity
		expectedUsage resource.Quantity
		expectedError bool
	}
	testCases := []testCase{
		{
			testName:      "memory-allowed",
			existing:      validPod("a", 1, getResourceRequirements(getResourceList("", "100Mi"), getResourceList("", ""))),
			input:         validPod("b", 1, getResourceRequirements(getResourceList("", "100Mi"), getResourceList("", ""))),
			resourceName:  api.ResourceMemory,
			hard:          resource.MustParse("500Mi"),
			expectedUsage: resource.MustParse("200Mi"),
			expectedError: false,
		},
		{
			testName:      "memory-not-allowed",
			existing:      validPod("a", 1, getResourceRequirements(getResourceList("", "100Mi"), getResourceList("", ""))),
			input:         validPod("b", 1, getResourceRequirements(getResourceList("", "450Mi"), getResourceList("", ""))),
			resourceName:  api.ResourceMemory,
			hard:          resource.MustParse("500Mi"),
			expectedError: true,
		},
		{
			testName:      "memory-not-allowed-with-different-format",
			existing:      validPod("a", 1, getResourceRequirements(getResourceList("", "100M"), getResourceList("", ""))),
			input:         validPod("b", 1, getResourceRequirements(getResourceList("", "450Mi"), getResourceList("", ""))),
			resourceName:  api.ResourceMemory,
			hard:          resource.MustParse("500Mi"),
			expectedError: true,
		},
		{
			testName:      "memory-no-request",
			existing:      validPod("a", 1, getResourceRequirements(getResourceList("", "100Mi"), getResourceList("", ""))),
			input:         validPod("b", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			resourceName:  api.ResourceMemory,
			hard:          resource.MustParse("500Mi"),
			expectedError: true,
		},
		{
			testName:      "cpu-allowed",
			existing:      validPod("a", 1, getResourceRequirements(getResourceList("1", ""), getResourceList("", ""))),
			input:         validPod("b", 1, getResourceRequirements(getResourceList("1", ""), getResourceList("", ""))),
			resourceName:  api.ResourceCPU,
			hard:          resource.MustParse("2"),
			expectedUsage: resource.MustParse("2"),
			expectedError: false,
		},
		{
			testName:      "cpu-not-allowed",
			existing:      validPod("a", 1, getResourceRequirements(getResourceList("1", ""), getResourceList("", ""))),
			input:         validPod("b", 1, getResourceRequirements(getResourceList("600m", ""), getResourceList("", ""))),
			resourceName:  api.ResourceCPU,
			hard:          resource.MustParse("1500m"),
			expectedError: true,
		},
		{
			testName:      "cpu-no-request",
			existing:      validPod("a", 1, getResourceRequirements(getResourceList("1", ""), getResourceList("", ""))),
			input:         validPod("b", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			resourceName:  api.ResourceCPU,
			hard:          resource.MustParse("1500m"),
			expectedError: true,
		},
	}
	for _, item := range testCases {
		podList := &api.PodList{Items: []api.Pod{*item.existing}}
		client := testclient.NewSimpleFake(podList)
		status := &api.ResourceQuotaStatus{
			Hard: api.ResourceList{},
			Used: api.ResourceList{},
		}
		used, err := resourcequotacontroller.PodRequests(item.existing, item.resourceName)
		if err != nil {
			t.Errorf("Test %s, unexpected error %v", item.testName, err)
		}
		status.Hard[item.resourceName] = item.hard
		status.Used[item.resourceName] = *used

		dirty, err := IncrementUsage(admission.NewAttributesRecord(item.input, "Pod", item.input.Namespace, item.input.Name, "pods", "", admission.Create, nil), status, client)
		if err == nil && item.expectedError {
			t.Errorf("Test %s, expected error", item.testName)
		}
		if err != nil && !item.expectedError {
			t.Errorf("Test %s, unexpected error", err)
		}
		if !item.expectedError {
			if !dirty {
				t.Errorf("Test %s, expected the quota to be dirty", item.testName)
			}
			quantity := status.Used[item.resourceName]
			if quantity.String() != item.expectedUsage.String() {
				t.Errorf("Test %s, expected usage %s, actual usage %s", item.testName, item.expectedUsage.String(), quantity.String())
			}
		}
	}
}

func TestIncrementUsagePods(t *testing.T) {
	pod := validPod("123", 1, getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", "")))
	podList := &api.PodList{Items: []api.Pod{*pod}}
	client := testclient.NewSimpleFake(podList)
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourcePods
	status.Hard[r] = resource.MustParse("2")
	status.Used[r] = resource.MustParse("1")
	dirty, err := IncrementUsage(admission.NewAttributesRecord(&api.Pod{}, "Pod", pod.Namespace, "new-pod", "pods", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	quantity := status.Used[r]
	if quantity.Value() != int64(2) {
		t.Errorf("Expected new item count to be 2, but was %s", quantity.String())
	}
}

func TestExceedUsagePods(t *testing.T) {
	pod := validPod("123", 1, getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", "")))
	podList := &api.PodList{Items: []api.Pod{*pod}}
	client := testclient.NewSimpleFake(podList)
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourcePods
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.Pod{}, "Pod", pod.Namespace, "name", "pods", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error because this would exceed your quota")
	}
}

func TestIncrementUsageServices(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceServices
	status.Hard[r] = resource.MustParse("2")
	status.Used[r] = resource.MustParse("1")
	dirty, err := IncrementUsage(admission.NewAttributesRecord(&api.Service{}, "Service", namespace, "name", "services", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	quantity := status.Used[r]
	if quantity.Value() != int64(2) {
		t.Errorf("Expected new item count to be 2, but was %s", quantity.String())
	}
}

func TestExceedUsageServices(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceServices
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.Service{}, "Service", namespace, "name", "services", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error because this would exceed usage")
	}
}

func TestIncrementUsageReplicationControllers(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ReplicationControllerList{
		Items: []api.ReplicationController{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceReplicationControllers
	status.Hard[r] = resource.MustParse("2")
	status.Used[r] = resource.MustParse("1")
	dirty, err := IncrementUsage(admission.NewAttributesRecord(&api.ReplicationController{}, "ReplicationController", namespace, "name", "replicationcontrollers", "", admission.Create, nil), status, client)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !dirty {
		t.Errorf("Expected the status to get incremented, therefore should have been dirty")
	}
	quantity := status.Used[r]
	if quantity.Value() != int64(2) {
		t.Errorf("Expected new item count to be 2, but was %s", quantity.String())
	}
}

func TestExceedUsageReplicationControllers(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.ReplicationControllerList{
		Items: []api.ReplicationController{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceReplicationControllers
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.ReplicationController{}, "ReplicationController", namespace, "name", "replicationcontrollers", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error for exceeding hard limits")
	}
}

func TestExceedUsageSecrets(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.SecretList{
		Items: []api.Secret{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourceSecrets
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.Secret{}, "Secret", namespace, "name", "secrets", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error for exceeding hard limits")
	}
}

func TestExceedUsagePersistentVolumeClaims(t *testing.T) {
	namespace := "default"
	client := testclient.NewSimpleFake(&api.PersistentVolumeClaimList{
		Items: []api.PersistentVolumeClaim{
			{
				ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
			},
		},
	})
	status := &api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	r := api.ResourcePersistentVolumeClaims
	status.Hard[r] = resource.MustParse("1")
	status.Used[r] = resource.MustParse("1")
	_, err := IncrementUsage(admission.NewAttributesRecord(&api.PersistentVolumeClaim{}, "PersistentVolumeClaim", namespace, "name", "persistentvolumeclaims", "", admission.Create, nil), status, client)
	if err == nil {
		t.Errorf("Expected error for exceeding hard limits")
	}
}
