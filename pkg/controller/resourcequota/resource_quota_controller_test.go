/*
Copyright 2015 The Kubernetes Authors.

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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/quota/install"
)

func getResourceList(cpu, memory string) v1.ResourceList {
	res := v1.ResourceList{}
	if cpu != "" {
		res[v1.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[v1.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func getResourceRequirements(requests, limits v1.ResourceList) v1.ResourceRequirements {
	res := v1.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func TestSyncResourceQuota(t *testing.T) {
	podList := v1.PodList{
		Items: []v1.Pod{
			{
				ObjectMeta: v1.ObjectMeta{Name: "pod-running", Namespace: "testing"},
				Status:     v1.PodStatus{Phase: v1.PodRunning},
				Spec: v1.PodSpec{
					Volumes:    []v1.Volume{{Name: "vol"}},
					Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
			{
				ObjectMeta: v1.ObjectMeta{Name: "pod-running-2", Namespace: "testing"},
				Status:     v1.PodStatus{Phase: v1.PodRunning},
				Spec: v1.PodSpec{
					Volumes:    []v1.Volume{{Name: "vol"}},
					Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
			{
				ObjectMeta: v1.ObjectMeta{Name: "pod-failed", Namespace: "testing"},
				Status:     v1.PodStatus{Phase: v1.PodFailed},
				Spec: v1.PodSpec{
					Volumes:    []v1.Volume{{Name: "vol"}},
					Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
		},
	}
	resourceQuota := v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{Name: "quota", Namespace: "testing"},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("3"),
				v1.ResourceMemory: resource.MustParse("100Gi"),
				v1.ResourcePods:   resource.MustParse("5"),
			},
		},
	}
	expectedUsage := v1.ResourceQuota{
		Status: v1.ResourceQuotaStatus{
			Hard: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("3"),
				v1.ResourceMemory: resource.MustParse("100Gi"),
				v1.ResourcePods:   resource.MustParse("5"),
			},
			Used: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("2Gi"),
				v1.ResourcePods:   resource.MustParse("2"),
			},
		},
	}

	kubeClient := fake.NewSimpleClientset(&podList, &resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient, nil),
		GroupKindsToReplenish: []schema.GroupKind{
			api.Kind("Pod"),
			api.Kind("Service"),
			api.Kind("ReplicationController"),
			api.Kind("PersistentVolumeClaim"),
		},
		ControllerFactory:         NewReplenishmentControllerFactoryFromClient(kubeClient),
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
	}
	quotaController := NewResourceQuotaController(resourceQuotaControllerOptions)
	err := quotaController.syncResourceQuota(resourceQuota)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expectedActionSet := sets.NewString(
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}

	lastActionIndex := len(kubeClient.Actions()) - 1
	usage := kubeClient.Actions()[lastActionIndex].(core.UpdateAction).GetObject().(*v1.ResourceQuota)

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
	resourceQuota := v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Namespace: "default",
			Name:      "rq",
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
		},
		Status: v1.ResourceQuotaStatus{
			Hard: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("3"),
			},
			Used: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("0"),
			},
		},
	}

	expectedUsage := v1.ResourceQuota{
		Status: v1.ResourceQuotaStatus{
			Hard: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
			Used: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("0"),
			},
		},
	}

	kubeClient := fake.NewSimpleClientset(&resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient, nil),
		GroupKindsToReplenish: []schema.GroupKind{
			api.Kind("Pod"),
			api.Kind("Service"),
			api.Kind("ReplicationController"),
			api.Kind("PersistentVolumeClaim"),
		},
		ControllerFactory:         NewReplenishmentControllerFactoryFromClient(kubeClient),
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
	}
	quotaController := NewResourceQuotaController(resourceQuotaControllerOptions)
	err := quotaController.syncResourceQuota(resourceQuota)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	expectedActionSet := sets.NewString(
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}

	lastActionIndex := len(kubeClient.Actions()) - 1
	usage := kubeClient.Actions()[lastActionIndex].(core.UpdateAction).GetObject().(*v1.ResourceQuota)

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
func TestSyncResourceQuotaSpecHardChange(t *testing.T) {
	resourceQuota := v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Namespace: "default",
			Name:      "rq",
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
		},
		Status: v1.ResourceQuotaStatus{
			Hard: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("3"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			Used: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("0"),
				v1.ResourceMemory: resource.MustParse("0"),
			},
		},
	}

	expectedUsage := v1.ResourceQuota{
		Status: v1.ResourceQuotaStatus{
			Hard: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
			Used: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("0"),
			},
		},
	}

	kubeClient := fake.NewSimpleClientset(&resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient, nil),
		GroupKindsToReplenish: []schema.GroupKind{
			api.Kind("Pod"),
			api.Kind("Service"),
			api.Kind("ReplicationController"),
			api.Kind("PersistentVolumeClaim"),
		},
		ControllerFactory:         NewReplenishmentControllerFactoryFromClient(kubeClient),
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
	}
	quotaController := NewResourceQuotaController(resourceQuotaControllerOptions)
	err := quotaController.syncResourceQuota(resourceQuota)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	expectedActionSet := sets.NewString(
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}

	lastActionIndex := len(kubeClient.Actions()) - 1
	usage := kubeClient.Actions()[lastActionIndex].(core.UpdateAction).GetObject().(*v1.ResourceQuota)

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

	// ensure usage hard and used are are synced with spec hard, not have dirty resource
	for k, v := range usage.Status.Hard {
		if k == v1.ResourceMemory {
			t.Errorf("Unexpected Usage Hard: Key: %v, Value: %v", k, v.String())
		}
	}

	for k, v := range usage.Status.Used {
		if k == v1.ResourceMemory {
			t.Errorf("Unexpected Usage Used: Key: %v, Value: %v", k, v.String())
		}
	}
}

func TestSyncResourceQuotaNoChange(t *testing.T) {
	resourceQuota := v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Namespace: "default",
			Name:      "rq",
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
		},
		Status: v1.ResourceQuotaStatus{
			Hard: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("4"),
			},
			Used: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("0"),
			},
		},
	}

	kubeClient := fake.NewSimpleClientset(&v1.PodList{}, &resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient, nil),
		GroupKindsToReplenish: []schema.GroupKind{
			api.Kind("Pod"),
			api.Kind("Service"),
			api.Kind("ReplicationController"),
			api.Kind("PersistentVolumeClaim"),
		},
		ControllerFactory:         NewReplenishmentControllerFactoryFromClient(kubeClient),
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
	}
	quotaController := NewResourceQuotaController(resourceQuotaControllerOptions)
	err := quotaController.syncResourceQuota(resourceQuota)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expectedActionSet := sets.NewString(
		strings.Join([]string{"list", "pods", ""}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}
}

func TestAddQuota(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient, nil),
		GroupKindsToReplenish: []schema.GroupKind{
			api.Kind("Pod"),
			api.Kind("ReplicationController"),
			api.Kind("PersistentVolumeClaim"),
		},
		ControllerFactory:         NewReplenishmentControllerFactoryFromClient(kubeClient),
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
	}
	quotaController := NewResourceQuotaController(resourceQuotaControllerOptions)

	delete(quotaController.registry.(*generic.GenericRegistry).InternalEvaluators, api.Kind("Service"))

	testCases := []struct {
		name string

		quota            *v1.ResourceQuota
		expectedPriority bool
	}{
		{
			name:             "no status",
			expectedPriority: true,
			quota: &v1.ResourceQuota{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("4"),
					},
				},
			},
		},
		{
			name:             "status, no usage",
			expectedPriority: true,
			quota: &v1.ResourceQuota{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("4"),
					},
				},
				Status: v1.ResourceQuotaStatus{
					Hard: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("4"),
					},
				},
			},
		},
		{
			name:             "status, mismatch",
			expectedPriority: true,
			quota: &v1.ResourceQuota{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("4"),
					},
				},
				Status: v1.ResourceQuotaStatus{
					Hard: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("6"),
					},
					Used: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("0"),
					},
				},
			},
		},
		{
			name:             "status, missing usage, but don't care",
			expectedPriority: false,
			quota: &v1.ResourceQuota{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceServices: resource.MustParse("4"),
					},
				},
				Status: v1.ResourceQuotaStatus{
					Hard: v1.ResourceList{
						v1.ResourceServices: resource.MustParse("4"),
					},
				},
			},
		},
		{
			name:             "ready",
			expectedPriority: false,
			quota: &v1.ResourceQuota{
				ObjectMeta: v1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("4"),
					},
				},
				Status: v1.ResourceQuotaStatus{
					Hard: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("4"),
					},
					Used: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("0"),
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		quotaController.addQuota(tc.quota)
		if tc.expectedPriority {
			if e, a := 1, quotaController.missingUsageQueue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
			if e, a := 0, quotaController.queue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
		} else {
			if e, a := 0, quotaController.missingUsageQueue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
			if e, a := 1, quotaController.queue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
		}

		for quotaController.missingUsageQueue.Len() > 0 {
			key, _ := quotaController.missingUsageQueue.Get()
			quotaController.missingUsageQueue.Done(key)
		}
		for quotaController.queue.Len() > 0 {
			key, _ := quotaController.queue.Get()
			quotaController.queue.Done(key)
		}
	}
}
