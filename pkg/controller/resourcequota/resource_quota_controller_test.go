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
	"fmt"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/quota"
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

func mockDiscoveryFunc() ([]*metav1.APIResourceList, error) {
	return []*metav1.APIResourceList{}, nil
}

func mockListerForResourceFunc(listersForResource map[schema.GroupVersionResource]cache.GenericLister) quota.ListerForResourceFunc {
	return func(gvr schema.GroupVersionResource) (cache.GenericLister, error) {
		lister, found := listersForResource[gvr]
		if !found {
			return nil, fmt.Errorf("no lister found for resource")
		}
		return lister, nil
	}
}

func newGenericLister(groupResource schema.GroupResource, items []runtime.Object) cache.GenericLister {
	store := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	for _, item := range items {
		store.Add(item)
	}
	return cache.NewGenericLister(store, groupResource)
}

type quotaController struct {
	*ResourceQuotaController
	stop chan struct{}
}

func setupQuotaController(t *testing.T, kubeClient kubernetes.Interface, lister quota.ListerForResourceFunc) quotaController {
	informerFactory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	quotaConfiguration := install.NewQuotaConfigurationForControllers(lister)
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		QuotaClient:               kubeClient.Core(),
		ResourceQuotaInformer:     informerFactory.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		IgnoredResourcesFunc:      quotaConfiguration.IgnoredResources,
		DiscoveryFunc:             mockDiscoveryFunc,
		Registry:                  generic.NewRegistry(quotaConfiguration.Evaluators()),
		InformersStarted:          alwaysStarted,
	}
	qc, err := NewResourceQuotaController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	go informerFactory.Start(stop)
	return quotaController{qc, stop}
}

func newTestPods() []runtime.Object {
	return []runtime.Object{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-running", Namespace: "testing"},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
			Spec: v1.PodSpec{
				Volumes:    []v1.Volume{{Name: "vol"}},
				Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
			},
		},
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-running-2", Namespace: "testing"},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
			Spec: v1.PodSpec{
				Volumes:    []v1.Volume{{Name: "vol"}},
				Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
			},
		},
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-failed", Namespace: "testing"},
			Status:     v1.PodStatus{Phase: v1.PodFailed},
			Spec: v1.PodSpec{
				Volumes:    []v1.Volume{{Name: "vol"}},
				Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
			},
		},
	}
}

func TestSyncResourceQuota(t *testing.T) {
	testCases := map[string]struct {
		gvr               schema.GroupVersionResource
		items             []runtime.Object
		quota             v1.ResourceQuota
		status            v1.ResourceQuotaStatus
		expectedActionSet sets.String
	}{
		"pods": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
				},
			},
			status: v1.ResourceQuotaStatus{
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
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPods(),
		},
		"quota-spec-hard-updated": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
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
			},
			status: v1.ResourceQuotaStatus{
				Hard: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("4"),
				},
				Used: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("0"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: []runtime.Object{},
		},
		"quota-unchanged": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
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
						v1.ResourceCPU: resource.MustParse("0"),
					},
				},
			},
			status: v1.ResourceQuotaStatus{
				Hard: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("4"),
				},
				Used: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("0"),
				},
			},
			expectedActionSet: sets.NewString(),
			items:             []runtime.Object{},
		},
	}

	for testName, testCase := range testCases {
		kubeClient := fake.NewSimpleClientset(&testCase.quota)
		listersForResourceConfig := map[schema.GroupVersionResource]cache.GenericLister{
			testCase.gvr: newGenericLister(testCase.gvr.GroupResource(), testCase.items),
		}
		qc := setupQuotaController(t, kubeClient, mockListerForResourceFunc(listersForResourceConfig))
		defer close(qc.stop)

		if err := qc.syncResourceQuota(&testCase.quota); err != nil {
			t.Fatalf("test: %s, unexpected error: %v", testName, err)
		}

		actionSet := sets.NewString()
		for _, action := range kubeClient.Actions() {
			actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
		}
		if !actionSet.HasAll(testCase.expectedActionSet.List()...) {
			t.Errorf("test: %s,\nExpected actions:\n%v\n but got:\n%v\nDifference:\n%v", testName, testCase.expectedActionSet, actionSet, testCase.expectedActionSet.Difference(actionSet))
		}

		lastActionIndex := len(kubeClient.Actions()) - 1
		usage := kubeClient.Actions()[lastActionIndex].(core.UpdateAction).GetObject().(*v1.ResourceQuota)

		// ensure usage is as expected
		if len(usage.Status.Hard) != len(testCase.status.Hard) {
			t.Errorf("test: %s, status hard lengths do not match", testName)
		}
		if len(usage.Status.Used) != len(testCase.status.Used) {
			t.Errorf("test: %s, status used lengths do not match", testName)
		}
		for k, v := range testCase.status.Hard {
			actual := usage.Status.Hard[k]
			actualValue := actual.String()
			expectedValue := v.String()
			if expectedValue != actualValue {
				t.Errorf("test: %s, Usage Hard: Key: %v, Expected: %v, Actual: %v", testName, k, expectedValue, actualValue)
			}
		}
		for k, v := range testCase.status.Used {
			actual := usage.Status.Used[k]
			actualValue := actual.String()
			expectedValue := v.String()
			if expectedValue != actualValue {
				t.Errorf("test: %s, Usage Used: Key: %v, Expected: %v, Actual: %v", testName, k, expectedValue, actualValue)
			}
		}
	}
}

func TestAddQuota(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	gvr := v1.SchemeGroupVersion.WithResource("pods")
	listersForResourceConfig := map[schema.GroupVersionResource]cache.GenericLister{
		gvr: newGenericLister(gvr.GroupResource(), newTestPods()),
	}

	qc := setupQuotaController(t, kubeClient, mockListerForResourceFunc(listersForResourceConfig))
	defer close(qc.stop)

	testCases := []struct {
		name             string
		quota            *v1.ResourceQuota
		expectedPriority bool
	}{
		{
			name:             "no status",
			expectedPriority: true,
			quota: &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
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
				ObjectMeta: metav1.ObjectMeta{
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
				ObjectMeta: metav1.ObjectMeta{
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
			name:             "status, missing usage, but don't care (no informer)",
			expectedPriority: false,
			quota: &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						"count/foobars.example.com": resource.MustParse("4"),
					},
				},
				Status: v1.ResourceQuotaStatus{
					Hard: v1.ResourceList{
						"count/foobars.example.com": resource.MustParse("4"),
					},
				},
			},
		},
		{
			name:             "ready",
			expectedPriority: false,
			quota: &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
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
		qc.addQuota(tc.quota)
		if tc.expectedPriority {
			if e, a := 1, qc.missingUsageQueue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
			if e, a := 0, qc.queue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
		} else {
			if e, a := 0, qc.missingUsageQueue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
			if e, a := 1, qc.queue.Len(); e != a {
				t.Errorf("%s: expected %v, got %v", tc.name, e, a)
			}
		}
		for qc.missingUsageQueue.Len() > 0 {
			key, _ := qc.missingUsageQueue.Get()
			qc.missingUsageQueue.Done(key)
		}
		for qc.queue.Len() > 0 {
			key, _ := qc.queue.Get()
			qc.queue.Done(key)
		}
	}
}
