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

package resourcequotacontroller

import (
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/util/sets"
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

func TestFilterQuotaPods(t *testing.T) {
	pods := []api.Pod{
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-running"},
			Status:     api.PodStatus{Phase: api.PodRunning},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-pending"},
			Status:     api.PodStatus{Phase: api.PodPending},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-succeeded"},
			Status:     api.PodStatus{Phase: api.PodSucceeded},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-unknown"},
			Status:     api.PodStatus{Phase: api.PodUnknown},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-failed"},
			Status:     api.PodStatus{Phase: api.PodFailed},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-failed-with-restart-always"},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
			},
			Status: api.PodStatus{Phase: api.PodFailed},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-failed-with-restart-on-failure"},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyOnFailure,
			},
			Status: api.PodStatus{Phase: api.PodFailed},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod-failed-with-restart-never"},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyNever,
			},
			Status: api.PodStatus{Phase: api.PodFailed},
		},
	}
	expectedResults := sets.NewString("pod-running",
		"pod-pending", "pod-unknown", "pod-failed-with-restart-always",
		"pod-failed-with-restart-on-failure")

	actualResults := sets.String{}
	result := FilterQuotaPods(pods)
	for i := range result {
		actualResults.Insert(result[i].Name)
	}

	if len(expectedResults) != len(actualResults) || !actualResults.HasAll(expectedResults.List()...) {
		t.Errorf("Expected results %v, Actual results %v", expectedResults, actualResults)
	}
}

func TestSyncResourceQuota(t *testing.T) {
	podList := api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "pod-running"},
				Status:     api.PodStatus{Phase: api.PodRunning},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "pod-running-2"},
				Status:     api.PodStatus{Phase: api.PodRunning},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "pod-failed"},
				Status:     api.PodStatus{Phase: api.PodFailed},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
		},
	}
	quota := api.ResourceQuota{
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
		},
	}
	expectedUsage := api.ResourceQuota{
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("2Gi"),
				api.ResourcePods:   resource.MustParse("2"),
			},
		},
	}

	kubeClient := testclient.NewSimpleFake(&podList, &quota)

	ResourceQuotaController := NewResourceQuotaController(kubeClient)
	err := ResourceQuotaController.syncResourceQuota(quota)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	usage := kubeClient.Actions()[1].(testclient.UpdateAction).GetObject().(*api.ResourceQuota)

	// ensure hard and used limits are what we expected
	for k, v := range expectedUsage.Status.Hard {
		actual := usage.Status.Hard[k]
		actualValue := actual.String()
		expectedValue := v.String()
		if expectedValue != actualValue {
			t.Errorf("Usage Hard: Key: %v, Expected: %v, Actual: %v", k, expectedValue, actualValue)
		}
	}
	for k, v := range expectedUsage.Status.Used {
		actual := usage.Status.Used[k]
		actualValue := actual.String()
		expectedValue := v.String()
		if expectedValue != actualValue {
			t.Errorf("Usage Used: Key: %v, Expected: %v, Actual: %v", k, expectedValue, actualValue)
		}
	}
}

func TestSyncResourceQuotaSpecChange(t *testing.T) {
	quota := api.ResourceQuota{
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU: resource.MustParse("4"),
			},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU: resource.MustParse("3"),
			},
			Used: api.ResourceList{
				api.ResourceCPU: resource.MustParse("0"),
			},
		},
	}

	expectedUsage := api.ResourceQuota{
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU: resource.MustParse("4"),
			},
			Used: api.ResourceList{
				api.ResourceCPU: resource.MustParse("0"),
			},
		},
	}

	kubeClient := testclient.NewSimpleFake(&quota)

	ResourceQuotaController := NewResourceQuotaController(kubeClient)
	err := ResourceQuotaController.syncResourceQuota(quota)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	usage := kubeClient.Actions()[1].(testclient.UpdateAction).GetObject().(*api.ResourceQuota)

	// ensure hard and used limits are what we expected
	for k, v := range expectedUsage.Status.Hard {
		actual := usage.Status.Hard[k]
		actualValue := actual.String()
		expectedValue := v.String()
		if expectedValue != actualValue {
			t.Errorf("Usage Hard: Key: %v, Expected: %v, Actual: %v", k, expectedValue, actualValue)
		}
	}
	for k, v := range expectedUsage.Status.Used {
		actual := usage.Status.Used[k]
		actualValue := actual.String()
		expectedValue := v.String()
		if expectedValue != actualValue {
			t.Errorf("Usage Used: Key: %v, Expected: %v, Actual: %v", k, expectedValue, actualValue)
		}
	}

}

func TestSyncResourceQuotaNoChange(t *testing.T) {
	quota := api.ResourceQuota{
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU: resource.MustParse("4"),
			},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU: resource.MustParse("4"),
			},
			Used: api.ResourceList{
				api.ResourceCPU: resource.MustParse("0"),
			},
		},
	}

	kubeClient := testclient.NewSimpleFake(&api.PodList{}, &quota)

	ResourceQuotaController := NewResourceQuotaController(kubeClient)
	err := ResourceQuotaController.syncResourceQuota(quota)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	actions := kubeClient.Actions()
	if len(actions) != 1 && !actions[0].Matches("list", "pods") {
		t.Errorf("SyncResourceQuota made an unexpected client action when state was not dirty: %v", kubeClient.Actions)
	}
}

func TestPodHasRequests(t *testing.T) {
	type testCase struct {
		pod            *api.Pod
		resourceName   api.ResourceName
		expectedResult bool
	}
	testCases := []testCase{
		{
			pod:            validPod("request-cpu", 2, getResourceRequirements(getResourceList("100m", ""), getResourceList("", ""))),
			resourceName:   api.ResourceCPU,
			expectedResult: true,
		},
		{
			pod:            validPod("no-request-cpu", 2, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			resourceName:   api.ResourceCPU,
			expectedResult: false,
		},
		{
			pod:            validPod("request-zero-cpu", 2, getResourceRequirements(getResourceList("0", ""), getResourceList("", ""))),
			resourceName:   api.ResourceCPU,
			expectedResult: false,
		},
		{
			pod:            validPod("request-memory", 2, getResourceRequirements(getResourceList("", "2Mi"), getResourceList("", ""))),
			resourceName:   api.ResourceMemory,
			expectedResult: true,
		},
		{
			pod:            validPod("no-request-memory", 2, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			resourceName:   api.ResourceMemory,
			expectedResult: false,
		},
		{
			pod:            validPod("request-zero-memory", 2, getResourceRequirements(getResourceList("", "0"), getResourceList("", ""))),
			resourceName:   api.ResourceMemory,
			expectedResult: false,
		},
	}
	for _, item := range testCases {
		if actual := PodHasRequests(item.pod, item.resourceName); item.expectedResult != actual {
			t.Errorf("Pod %s for resource %s expected %v actual %v", item.pod.Name, item.resourceName, item.expectedResult, actual)
		}
	}
}

func TestPodRequests(t *testing.T) {
	type testCase struct {
		pod            *api.Pod
		resourceName   api.ResourceName
		expectedResult string
		expectedError  bool
	}
	testCases := []testCase{
		{
			pod:            validPod("request-cpu", 2, getResourceRequirements(getResourceList("100m", ""), getResourceList("", ""))),
			resourceName:   api.ResourceCPU,
			expectedResult: "200m",
			expectedError:  false,
		},
		{
			pod:            validPod("no-request-cpu", 2, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			resourceName:   api.ResourceCPU,
			expectedResult: "",
			expectedError:  true,
		},
		{
			pod:            validPod("request-zero-cpu", 2, getResourceRequirements(getResourceList("0", ""), getResourceList("", ""))),
			resourceName:   api.ResourceCPU,
			expectedResult: "",
			expectedError:  true,
		},
		{
			pod:            validPod("request-memory", 2, getResourceRequirements(getResourceList("", "500Mi"), getResourceList("", ""))),
			resourceName:   api.ResourceMemory,
			expectedResult: "1000Mi",
			expectedError:  false,
		},
		{
			pod:            validPod("no-request-memory", 2, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			resourceName:   api.ResourceMemory,
			expectedResult: "",
			expectedError:  true,
		},
		{
			pod:            validPod("request-zero-memory", 2, getResourceRequirements(getResourceList("", "0"), getResourceList("", ""))),
			resourceName:   api.ResourceMemory,
			expectedResult: "",
			expectedError:  true,
		},
	}
	for _, item := range testCases {
		actual, err := PodRequests(item.pod, item.resourceName)
		if item.expectedError != (err != nil) {
			t.Errorf("Unexpected error result for pod %s for resource %s expected error %v got %v", item.pod.Name, item.resourceName, item.expectedError, err)
		}
		if item.expectedResult != "" && (item.expectedResult != actual.String()) {
			t.Errorf("Expected %s, Actual %s, pod %s for resource %s", item.expectedResult, actual.String(), item.pod.Name, item.resourceName)
		}
	}
}

func TestPodsRequests(t *testing.T) {
	type testCase struct {
		pods           []*api.Pod
		resourceName   api.ResourceName
		expectedResult string
	}
	testCases := []testCase{
		{
			pods: []*api.Pod{
				validPod("request-cpu-1", 1, getResourceRequirements(getResourceList("100m", ""), getResourceList("", ""))),
				validPod("request-cpu-2", 1, getResourceRequirements(getResourceList("1", ""), getResourceList("", ""))),
			},
			resourceName:   api.ResourceCPU,
			expectedResult: "1100m",
		},
		{
			pods: []*api.Pod{
				validPod("no-request-cpu-1", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
				validPod("no-request-cpu-2", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			},
			resourceName:   api.ResourceCPU,
			expectedResult: "",
		},
		{
			pods: []*api.Pod{
				validPod("request-zero-cpu-1", 1, getResourceRequirements(getResourceList("0", ""), getResourceList("", ""))),
				validPod("request-zero-cpu-1", 1, getResourceRequirements(getResourceList("0", ""), getResourceList("", ""))),
			},
			resourceName:   api.ResourceCPU,
			expectedResult: "",
		},
		{
			pods: []*api.Pod{
				validPod("request-memory-1", 1, getResourceRequirements(getResourceList("", "500Mi"), getResourceList("", ""))),
				validPod("request-memory-2", 1, getResourceRequirements(getResourceList("", "1Gi"), getResourceList("", ""))),
			},
			resourceName:   api.ResourceMemory,
			expectedResult: "1524Mi",
		},
	}
	for _, item := range testCases {
		actual := PodsRequests(item.pods, item.resourceName)
		if item.expectedResult != "" && (item.expectedResult != actual.String()) {
			t.Errorf("Expected %s, Actual %s, pod %s for resource %s", item.expectedResult, actual.String(), item.pods[0].Name, item.resourceName)
		}
	}
}
