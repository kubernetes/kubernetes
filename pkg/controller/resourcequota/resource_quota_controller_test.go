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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/quota/install"
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

func TestSyncResourceQuota(t *testing.T) {
	podList := api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "pod-running", Namespace: "testing"},
				Status:     api.PodStatus{Phase: api.PodRunning},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "pod-running-2", Namespace: "testing"},
				Status:     api.PodStatus{Phase: api.PodRunning},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "pod-failed", Namespace: "testing"},
				Status:     api.PodStatus{Phase: api.PodFailed},
				Spec: api.PodSpec{
					Volumes:    []api.Volume{{Name: "vol"}},
					Containers: []api.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				},
			},
		},
	}
	resourceQuota := api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "testing"},
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

	kubeClient := fake.NewSimpleClientset(&podList, &resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient),
		GroupKindsToReplenish: []unversioned.GroupKind{
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
	usage := kubeClient.Actions()[lastActionIndex].(core.UpdateAction).GetObject().(*api.ResourceQuota)

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
	resourceQuota := api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Namespace: "default",
			Name:      "rq",
		},
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

	kubeClient := fake.NewSimpleClientset(&resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient),
		GroupKindsToReplenish: []unversioned.GroupKind{
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
	usage := kubeClient.Actions()[lastActionIndex].(core.UpdateAction).GetObject().(*api.ResourceQuota)

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
	resourceQuota := api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Namespace: "default",
			Name:      "rq",
		},
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU: resource.MustParse("4"),
			},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("0"),
				api.ResourceMemory: resource.MustParse("0"),
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

	kubeClient := fake.NewSimpleClientset(&resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient),
		GroupKindsToReplenish: []unversioned.GroupKind{
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
	usage := kubeClient.Actions()[lastActionIndex].(core.UpdateAction).GetObject().(*api.ResourceQuota)

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
		if k == api.ResourceMemory {
			t.Errorf("Unexpected Usage Hard: Key: %v, Value: %v", k, v.String())
		}
	}

	for k, v := range usage.Status.Used {
		if k == api.ResourceMemory {
			t.Errorf("Unexpected Usage Used: Key: %v, Value: %v", k, v.String())
		}
	}
}

func TestSyncResourceQuotaNoChange(t *testing.T) {
	resourceQuota := api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Namespace: "default",
			Name:      "rq",
		},
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

	kubeClient := fake.NewSimpleClientset(&api.PodList{}, &resourceQuota)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		KubeClient:   kubeClient,
		ResyncPeriod: controller.NoResyncPeriodFunc,
		Registry:     install.NewRegistry(kubeClient),
		GroupKindsToReplenish: []unversioned.GroupKind{
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
		Registry:     install.NewRegistry(kubeClient),
		GroupKindsToReplenish: []unversioned.GroupKind{
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

		quota            *api.ResourceQuota
		expectedPriority bool
	}{
		{
			name:             "no status",
			expectedPriority: true,
			quota: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: api.ResourceQuotaSpec{
					Hard: api.ResourceList{
						api.ResourceCPU: resource.MustParse("4"),
					},
				},
			},
		},
		{
			name:             "status, no usage",
			expectedPriority: true,
			quota: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: api.ResourceQuotaSpec{
					Hard: api.ResourceList{
						api.ResourceCPU: resource.MustParse("4"),
					},
				},
				Status: api.ResourceQuotaStatus{
					Hard: api.ResourceList{
						api.ResourceCPU: resource.MustParse("4"),
					},
				},
			},
		},
		{
			name:             "status, mismatch",
			expectedPriority: true,
			quota: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: api.ResourceQuotaSpec{
					Hard: api.ResourceList{
						api.ResourceCPU: resource.MustParse("4"),
					},
				},
				Status: api.ResourceQuotaStatus{
					Hard: api.ResourceList{
						api.ResourceCPU: resource.MustParse("6"),
					},
					Used: api.ResourceList{
						api.ResourceCPU: resource.MustParse("0"),
					},
				},
			},
		},
		{
			name:             "status, missing usage, but don't care",
			expectedPriority: false,
			quota: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: api.ResourceQuotaSpec{
					Hard: api.ResourceList{
						api.ResourceServices: resource.MustParse("4"),
					},
				},
				Status: api.ResourceQuotaStatus{
					Hard: api.ResourceList{
						api.ResourceServices: resource.MustParse("4"),
					},
				},
			},
		},
		{
			name:             "ready",
			expectedPriority: false,
			quota: &api.ResourceQuota{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
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
