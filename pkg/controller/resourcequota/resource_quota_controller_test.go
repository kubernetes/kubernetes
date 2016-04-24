/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
		strings.Join([]string{"list", "replicationcontrollers", ""}, "-"),
		strings.Join([]string{"list", "services", ""}, "-"),
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"list", "resourcequotas", ""}, "-"),
		strings.Join([]string{"list", "secrets", ""}, "-"),
		strings.Join([]string{"list", "persistentvolumeclaims", ""}, "-"),
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
		strings.Join([]string{"list", "replicationcontrollers", ""}, "-"),
		strings.Join([]string{"list", "services", ""}, "-"),
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"list", "resourcequotas", ""}, "-"),
		strings.Join([]string{"list", "secrets", ""}, "-"),
		strings.Join([]string{"list", "persistentvolumeclaims", ""}, "-"),
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

func TestSyncResourceQuotaNoChange(t *testing.T) {
	resourceQuota := api.ResourceQuota{
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
		strings.Join([]string{"list", "replicationcontrollers", ""}, "-"),
		strings.Join([]string{"list", "services", ""}, "-"),
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"list", "resourcequotas", ""}, "-"),
		strings.Join([]string{"list", "secrets", ""}, "-"),
		strings.Join([]string{"list", "persistentvolumeclaims", ""}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}
}
