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
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/quota/v1"
	"k8s.io/kubernetes/pkg/quota/v1/generic"
	"k8s.io/kubernetes/pkg/quota/v1/install"
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

func newErrorLister() cache.GenericLister {
	return errorLister{}
}

type errorLister struct {
}

func (errorLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	return nil, fmt.Errorf("error listing")
}
func (errorLister) Get(name string) (runtime.Object, error) {
	return nil, fmt.Errorf("error getting")
}
func (errorLister) ByNamespace(namespace string) cache.GenericNamespaceLister {
	return errorLister{}
}

type quotaController struct {
	*Controller
	stop chan struct{}
}

func setupQuotaController(t *testing.T, kubeClient kubernetes.Interface, lister quota.ListerForResourceFunc, discoveryFunc NamespacedResourcesFunc) quotaController {
	informerFactory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	quotaConfiguration := install.NewQuotaConfigurationForControllers(lister)
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)
	resourceQuotaControllerOptions := &ControllerOptions{
		QuotaClient:               kubeClient.CoreV1(),
		ResourceQuotaInformer:     informerFactory.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		IgnoredResourcesFunc:      quotaConfiguration.IgnoredResources,
		DiscoveryFunc:             discoveryFunc,
		Registry:                  generic.NewRegistry(quotaConfiguration.Evaluators()),
		InformersStarted:          alwaysStarted,
		InformerFactory:           informerFactory,
	}
	qc, err := NewController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	informerFactory.Start(stop)
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

func newBestEffortTestPods() []runtime.Object {
	return []runtime.Object{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-running", Namespace: "testing"},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
			Spec: v1.PodSpec{
				Volumes:    []v1.Volume{{Name: "vol"}},
				Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("", ""), getResourceList("", ""))}},
			},
		},
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-running-2", Namespace: "testing"},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
			Spec: v1.PodSpec{
				Volumes:    []v1.Volume{{Name: "vol"}},
				Containers: []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("", ""), getResourceList("", ""))}},
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

func newTestPodsWithPriorityClasses() []runtime.Object {
	return []runtime.Object{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-running", Namespace: "testing"},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
			Spec: v1.PodSpec{
				Volumes:           []v1.Volume{{Name: "vol"}},
				Containers:        []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("500m", "50Gi"), getResourceList("", ""))}},
				PriorityClassName: "high",
			},
		},
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-running-2", Namespace: "testing"},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
			Spec: v1.PodSpec{
				Volumes:           []v1.Volume{{Name: "vol"}},
				Containers:        []v1.Container{{Name: "ctr", Image: "image", Resources: getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", ""))}},
				PriorityClassName: "low",
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
		errorGVR          schema.GroupVersionResource
		items             []runtime.Object
		quota             v1.ResourceQuota
		status            v1.ResourceQuotaStatus
		expectedError     string
		expectedActionSet sets.String
	}{
		"non-matching-best-effort-scoped-quota": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					Scopes: []v1.ResourceQuotaScope{v1.ResourceQuotaScopeBestEffort},
				},
			},
			status: v1.ResourceQuotaStatus{
				Hard: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("3"),
					v1.ResourceMemory: resource.MustParse("100Gi"),
					v1.ResourcePods:   resource.MustParse("5"),
				},
				Used: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("0"),
					v1.ResourceMemory: resource.MustParse("0"),
					v1.ResourcePods:   resource.MustParse("0"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPods(),
		},
		"matching-best-effort-scoped-quota": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					Scopes: []v1.ResourceQuotaScope{v1.ResourceQuotaScopeBestEffort},
				},
			},
			status: v1.ResourceQuotaStatus{
				Hard: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("3"),
					v1.ResourceMemory: resource.MustParse("100Gi"),
					v1.ResourcePods:   resource.MustParse("5"),
				},
				Used: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("0"),
					v1.ResourceMemory: resource.MustParse("0"),
					v1.ResourcePods:   resource.MustParse("2"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newBestEffortTestPods(),
		},
		"non-matching-priorityclass-scoped-quota-OpExists": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpExists},
						},
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
					v1.ResourceCPU:    resource.MustParse("0"),
					v1.ResourceMemory: resource.MustParse("0"),
					v1.ResourcePods:   resource.MustParse("0"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPods(),
		},
		"matching-priorityclass-scoped-quota-OpExists": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpExists},
						},
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
					v1.ResourceCPU:    resource.MustParse("600m"),
					v1.ResourceMemory: resource.MustParse("51Gi"),
					v1.ResourcePods:   resource.MustParse("2"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPodsWithPriorityClasses(),
		},
		"matching-priorityclass-scoped-quota-OpIn": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpIn,
								Values:    []string{"high", "low"},
							},
						},
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
					v1.ResourceCPU:    resource.MustParse("600m"),
					v1.ResourceMemory: resource.MustParse("51Gi"),
					v1.ResourcePods:   resource.MustParse("2"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPodsWithPriorityClasses(),
		},
		"matching-priorityclass-scoped-quota-OpIn-high": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpIn,
								Values:    []string{"high"},
							},
						},
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
					v1.ResourceCPU:    resource.MustParse("500m"),
					v1.ResourceMemory: resource.MustParse("50Gi"),
					v1.ResourcePods:   resource.MustParse("1"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPodsWithPriorityClasses(),
		},
		"matching-priorityclass-scoped-quota-OpIn-low": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpIn,
								Values:    []string{"low"},
							},
						},
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
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
					v1.ResourcePods:   resource.MustParse("1"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPodsWithPriorityClasses(),
		},
		"matching-priorityclass-scoped-quota-OpNotIn-low": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpNotIn,
								Values:    []string{"high"},
							},
						},
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
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
					v1.ResourcePods:   resource.MustParse("1"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPodsWithPriorityClasses(),
		},
		"non-matching-priorityclass-scoped-quota-OpIn": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpIn,
								Values:    []string{"random"},
							},
						},
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
					v1.ResourceCPU:    resource.MustParse("0"),
					v1.ResourceMemory: resource.MustParse("0"),
					v1.ResourcePods:   resource.MustParse("0"),
				},
			},
			expectedActionSet: sets.NewString(
				strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
			),
			items: newTestPodsWithPriorityClasses(),
		},
		"non-matching-priorityclass-scoped-quota-OpNotIn": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpNotIn,
								Values:    []string{"random"},
							},
						},
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
		"matching-priorityclass-scoped-quota-OpDoesNotExist": {
			gvr: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "testing"},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3"),
						v1.ResourceMemory: resource.MustParse("100Gi"),
						v1.ResourcePods:   resource.MustParse("5"),
					},
					ScopeSelector: &v1.ScopeSelector{
						MatchExpressions: []v1.ScopedResourceSelectorRequirement{
							{
								ScopeName: v1.ResourceQuotaScopePriorityClass,
								Operator:  v1.ScopeSelectorOpDoesNotExist,
							},
						},
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
		"quota-missing-status-with-calculation-error": {
			errorGVR: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourcePods: resource.MustParse("1"),
					},
				},
				Status: v1.ResourceQuotaStatus{},
			},
			status: v1.ResourceQuotaStatus{
				Hard: v1.ResourceList{
					v1.ResourcePods: resource.MustParse("1"),
				},
			},
			expectedError:     "error listing",
			expectedActionSet: sets.NewString("update-resourcequotas-status"),
			items:             []runtime.Object{},
		},
		"quota-missing-status-with-partial-calculation-error": {
			gvr:      v1.SchemeGroupVersion.WithResource("configmaps"),
			errorGVR: v1.SchemeGroupVersion.WithResource("pods"),
			quota: v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						v1.ResourcePods:       resource.MustParse("1"),
						v1.ResourceConfigMaps: resource.MustParse("1"),
					},
				},
				Status: v1.ResourceQuotaStatus{},
			},
			status: v1.ResourceQuotaStatus{
				Hard: v1.ResourceList{
					v1.ResourcePods:       resource.MustParse("1"),
					v1.ResourceConfigMaps: resource.MustParse("1"),
				},
				Used: v1.ResourceList{
					v1.ResourceConfigMaps: resource.MustParse("0"),
				},
			},
			expectedError:     "error listing",
			expectedActionSet: sets.NewString("update-resourcequotas-status"),
			items:             []runtime.Object{},
		},
	}

	for testName, testCase := range testCases {
		kubeClient := fake.NewSimpleClientset(&testCase.quota)
		listersForResourceConfig := map[schema.GroupVersionResource]cache.GenericLister{
			testCase.gvr:      newGenericLister(testCase.gvr.GroupResource(), testCase.items),
			testCase.errorGVR: newErrorLister(),
		}
		qc := setupQuotaController(t, kubeClient, mockListerForResourceFunc(listersForResourceConfig), mockDiscoveryFunc)
		defer close(qc.stop)

		if err := qc.syncResourceQuota(&testCase.quota); err != nil {
			if len(testCase.expectedError) == 0 || !strings.Contains(err.Error(), testCase.expectedError) {
				t.Fatalf("test: %s, unexpected error: %v", testName, err)
			}
		} else if len(testCase.expectedError) > 0 {
			t.Fatalf("test: %s, expected error %q, got none", testName, testCase.expectedError)
		}

		actionSet := sets.NewString()
		for _, action := range kubeClient.Actions() {
			actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
		}
		if !actionSet.HasAll(testCase.expectedActionSet.List()...) {
			t.Errorf("test: %s,\nExpected actions:\n%v\n but got:\n%v\nDifference:\n%v", testName, testCase.expectedActionSet, actionSet, testCase.expectedActionSet.Difference(actionSet))
		}

		var usage *v1.ResourceQuota
		actions := kubeClient.Actions()
		for i := len(actions) - 1; i >= 0; i-- {
			if updateAction, ok := actions[i].(core.UpdateAction); ok {
				usage = updateAction.GetObject().(*v1.ResourceQuota)
				break
			}
		}
		if usage == nil {
			t.Fatalf("test: %s,\nExpected update action usage, got none: actions:\n%v", testName, actions)
		}

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

	qc := setupQuotaController(t, kubeClient, mockListerForResourceFunc(listersForResourceConfig), mockDiscoveryFunc)
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
			name:             "status, no usage(to validate it works for extended resources)",
			expectedPriority: true,
			quota: &v1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "rq",
				},
				Spec: v1.ResourceQuotaSpec{
					Hard: v1.ResourceList{
						"requests.example/foobars.example.com": resource.MustParse("4"),
					},
				},
				Status: v1.ResourceQuotaStatus{
					Hard: v1.ResourceList{
						"requests.example/foobars.example.com": resource.MustParse("4"),
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
						"foobars.example.com": resource.MustParse("4"),
					},
				},
				Status: v1.ResourceQuotaStatus{
					Hard: v1.ResourceList{
						"foobars.example.com": resource.MustParse("4"),
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

// TestDiscoverySync ensures that a discovery client error
// will not cause the quota controller to block infinitely.
func TestDiscoverySync(t *testing.T) {
	serverResources := []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: metav1.Verbs{"create", "delete", "list", "watch"}},
			},
		},
	}
	unsyncableServerResources := []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: metav1.Verbs{"create", "delete", "list", "watch"}},
				{Name: "secrets", Namespaced: true, Kind: "Secret", Verbs: metav1.Verbs{"create", "delete", "list", "watch"}},
			},
		},
	}
	fakeDiscoveryClient := &fakeServerResources{
		PreferredResources: serverResources,
		Error:              nil,
		Lock:               sync.Mutex{},
		InterfaceUsedCount: 0,
	}

	testHandler := &fakeActionHandler{
		response: map[string]FakeResponse{
			"GET" + "/api/v1/pods": {
				200,
				[]byte("{}"),
			},
			"GET" + "/api/v1/secrets": {
				404,
				[]byte("{}"),
			},
		},
	}

	srv, clientConfig := testServerAndClientConfig(testHandler.ServeHTTP)
	defer srv.Close()
	clientConfig.ContentConfig.NegotiatedSerializer = nil
	kubeClient, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	pods := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	secrets := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "secrets"}
	listersForResourceConfig := map[schema.GroupVersionResource]cache.GenericLister{
		pods:    newGenericLister(pods.GroupResource(), []runtime.Object{}),
		secrets: newGenericLister(secrets.GroupResource(), []runtime.Object{}),
	}
	qc := setupQuotaController(t, kubeClient, mockListerForResourceFunc(listersForResourceConfig), fakeDiscoveryClient.ServerPreferredNamespacedResources)
	defer close(qc.stop)

	stopSync := make(chan struct{})
	defer close(stopSync)
	// The pseudo-code of Sync():
	// Sync(client, period, stopCh):
	//    wait.Until() loops with `period` until the `stopCh` is closed :
	//       GetQuotableResources()
	//       resyncMonitors()
	//       cache.WaitForNamedCacheSync() loops with `syncedPollPeriod` (hardcoded to 100ms), until either its stop channel is closed after `period`, or all caches synced.
	//
	// Setting the period to 200ms allows the WaitForCacheSync() to check
	// for cache sync ~2 times in every wait.Until() loop.
	//
	// The 1s sleep in the test allows GetQuotableResources and
	// resyncMonitors to run ~5 times to ensure the changes to the
	// fakeDiscoveryClient are picked up.
	go qc.Sync(fakeDiscoveryClient.ServerPreferredNamespacedResources, 200*time.Millisecond, stopSync)

	// Wait until the sync discovers the initial resources
	time.Sleep(1 * time.Second)

	err = expectSyncNotBlocked(fakeDiscoveryClient, &qc.workerLock)
	if err != nil {
		t.Fatalf("Expected quotacontroller.Sync to be running but it is blocked: %v", err)
	}

	// Simulate the discovery client returning an error
	fakeDiscoveryClient.setPreferredResources(nil)
	fakeDiscoveryClient.setError(fmt.Errorf("error calling discoveryClient.ServerPreferredResources()"))

	// Wait until sync discovers the change
	time.Sleep(1 * time.Second)

	// Remove the error from being returned and see if the quota sync is still working
	fakeDiscoveryClient.setPreferredResources(serverResources)
	fakeDiscoveryClient.setError(nil)

	err = expectSyncNotBlocked(fakeDiscoveryClient, &qc.workerLock)
	if err != nil {
		t.Fatalf("Expected quotacontroller.Sync to still be running but it is blocked: %v", err)
	}

	// Simulate the discovery client returning a resource the restmapper can resolve, but will not sync caches
	fakeDiscoveryClient.setPreferredResources(unsyncableServerResources)
	fakeDiscoveryClient.setError(nil)

	// Wait until sync discovers the change
	time.Sleep(1 * time.Second)

	// Put the resources back to normal and ensure quota sync recovers
	fakeDiscoveryClient.setPreferredResources(serverResources)
	fakeDiscoveryClient.setError(nil)

	err = expectSyncNotBlocked(fakeDiscoveryClient, &qc.workerLock)
	if err != nil {
		t.Fatalf("Expected quotacontroller.Sync to still be running but it is blocked: %v", err)
	}
}

// testServerAndClientConfig returns a server that listens and a config that can reference it
func testServerAndClientConfig(handler func(http.ResponseWriter, *http.Request)) (*httptest.Server, *rest.Config) {
	srv := httptest.NewServer(http.HandlerFunc(handler))
	config := &rest.Config{
		Host: srv.URL,
	}
	return srv, config
}

func expectSyncNotBlocked(fakeDiscoveryClient *fakeServerResources, workerLock *sync.RWMutex) error {
	before := fakeDiscoveryClient.getInterfaceUsedCount()
	t := 1 * time.Second
	time.Sleep(t)
	after := fakeDiscoveryClient.getInterfaceUsedCount()
	if before == after {
		return fmt.Errorf("discoveryClient.ServerPreferredResources() called %d times over %v", after-before, t)
	}

	workerLockAcquired := make(chan struct{})
	go func() {
		workerLock.Lock()
		defer workerLock.Unlock()
		close(workerLockAcquired)
	}()
	select {
	case <-workerLockAcquired:
		return nil
	case <-time.After(t):
		return fmt.Errorf("workerLock blocked for at least %v", t)
	}
}

type fakeServerResources struct {
	PreferredResources []*metav1.APIResourceList
	Error              error
	Lock               sync.Mutex
	InterfaceUsedCount int
}

func (*fakeServerResources) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	return nil, nil
}

func (*fakeServerResources) ServerResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

func (*fakeServerResources) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

func (f *fakeServerResources) setPreferredResources(resources []*metav1.APIResourceList) {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.PreferredResources = resources
}

func (f *fakeServerResources) setError(err error) {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.Error = err
}

func (f *fakeServerResources) getInterfaceUsedCount() int {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	return f.InterfaceUsedCount
}

func (f *fakeServerResources) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.InterfaceUsedCount++
	return f.PreferredResources, f.Error
}

// fakeAction records information about requests to aid in testing.
type fakeAction struct {
	method string
	path   string
	query  string
}

// String returns method=path to aid in testing
func (f *fakeAction) String() string {
	return strings.Join([]string{f.method, f.path}, "=")
}

type FakeResponse struct {
	statusCode int
	content    []byte
}

// fakeActionHandler holds a list of fakeActions received
type fakeActionHandler struct {
	// statusCode and content returned by this handler for different method + path.
	response map[string]FakeResponse

	lock    sync.Mutex
	actions []fakeAction
}

// ServeHTTP logs the action that occurred and always returns the associated status code
func (f *fakeActionHandler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	func() {
		f.lock.Lock()
		defer f.lock.Unlock()

		f.actions = append(f.actions, fakeAction{method: request.Method, path: request.URL.Path, query: request.URL.RawQuery})
		fakeResponse, ok := f.response[request.Method+request.URL.Path]
		if !ok {
			fakeResponse.statusCode = 200
			fakeResponse.content = []byte("{\"kind\": \"List\"}")
		}
		response.Header().Set("Content-Type", "application/json")
		response.WriteHeader(fakeResponse.statusCode)
		response.Write(fakeResponse.content)
	}()

	// This is to allow the fakeActionHandler to simulate a watch being opened
	if strings.Contains(request.URL.RawQuery, "watch=true") {
		hijacker, ok := response.(http.Hijacker)
		if !ok {
			return
		}
		connection, _, err := hijacker.Hijack()
		if err != nil {
			return
		}
		defer connection.Close()
		time.Sleep(30 * time.Second)
	}
}
