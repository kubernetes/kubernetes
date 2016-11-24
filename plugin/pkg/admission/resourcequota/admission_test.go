/*
Copyright 2014 The Kubernetes Authors.

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
	"strings"
	"testing"
	"time"

	lru "github.com/hashicorp/golang-lru"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	testcore "k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/evaluator/core"
	"k8s.io/kubernetes/pkg/quota/generic"
	"k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
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

func validPersistentVolumeClaim(name string, resources api.ResourceRequirements) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "test"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: resources,
		},
	}
}

func TestPrettyPrint(t *testing.T) {
	toResourceList := func(resources map[api.ResourceName]string) api.ResourceList {
		resourceList := api.ResourceList{}
		for key, value := range resources {
			resourceList[key] = resource.MustParse(value)
		}
		return resourceList
	}
	testCases := []struct {
		input    api.ResourceList
		expected string
	}{
		{
			input: toResourceList(map[api.ResourceName]string{
				api.ResourceCPU: "100m",
			}),
			expected: "cpu=100m",
		},
		{
			input: toResourceList(map[api.ResourceName]string{
				api.ResourcePods:                   "10",
				api.ResourceServices:               "10",
				api.ResourceReplicationControllers: "10",
				api.ResourceServicesNodePorts:      "10",
				api.ResourceRequestsCPU:            "100m",
				api.ResourceRequestsMemory:         "100Mi",
				api.ResourceLimitsCPU:              "100m",
				api.ResourceLimitsMemory:           "100Mi",
			}),
			expected: "limits.cpu=100m,limits.memory=100Mi,pods=10,replicationcontrollers=10,requests.cpu=100m,requests.memory=100Mi,services=10,services.nodeports=10",
		},
	}
	for i, testCase := range testCases {
		result := prettyPrint(testCase.input)
		if result != testCase.expected {
			t.Errorf("Pretty print did not give stable sorted output[%d], expected %v, but got %v", i, testCase.expected, result)
		}
	}
}

// TestAdmissionIgnoresDelete verifies that the admission controller ignores delete operations
func TestAdmissionIgnoresDelete(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	handler, err := NewResourceQuota(kubeClient, install.NewRegistry(nil, nil), 5, stopCh)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	namespace := "default"
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Pod").WithVersion("version"), namespace, "name", api.Resource("pods").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("ResourceQuota should admit all deletes: %v", err)
	}
}

// TestAdmissionIgnoresSubresources verifies that the admission controller ignores subresources
// It verifies that creation of a pod that would have exceeded quota is properly failed
// It verifies that create operations to a subresource that would have exceeded quota would succeed
func TestAdmissionIgnoresSubresources(t *testing.T) {
	resourceQuota := &api.ResourceQuota{}
	resourceQuota.Name = "quota"
	resourceQuota.Namespace = "test"
	resourceQuota.Status = api.ResourceQuotaStatus{
		Hard: api.ResourceList{},
		Used: api.ResourceList{},
	}
	resourceQuota.Status.Hard[api.ResourceMemory] = resource.MustParse("2Gi")
	resourceQuota.Status.Used[api.ResourceMemory] = resource.MustParse("1Gi")
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)
	newPod := validPod("123", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected an error because the pod exceeded allowed quota")
	}
	err = handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "subresource", admission.Create, nil))
	if err != nil {
		t.Errorf("Did not expect an error because the action went to a subresource: %v", err)
	}
}

// TestAdmitBelowQuotaLimit verifies that a pod when created has its usage reflected on the quota
func TestAdmitBelowQuotaLimit(t *testing.T) {
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1"),
				api.ResourceMemory: resource.MustParse("50Gi"),
				api.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(kubeClient.Actions()) == 0 {
		t.Errorf("Expected a client action")
	}

	expectedActionSet := sets.NewString(
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}

	decimatedActions := removeListWatch(kubeClient.Actions())
	lastActionIndex := len(decimatedActions) - 1
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*api.ResourceQuota)
	expectedUsage := api.ResourceQuota{
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1100m"),
				api.ResourceMemory: resource.MustParse("52Gi"),
				api.ResourcePods:   resource.MustParse("4"),
			},
		},
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

// TestAdmitHandlesOldObjects verifies that admit handles updates correctly with old objects
func TestAdmitHandlesOldObjects(t *testing.T) {
	// in this scenario, the old quota was based on a service type=loadbalancer
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceServices:              resource.MustParse("10"),
				api.ResourceServicesLoadBalancers: resource.MustParse("10"),
				api.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: api.ResourceList{
				api.ResourceServices:              resource.MustParse("1"),
				api.ResourceServicesLoadBalancers: resource.MustParse("1"),
				api.ResourceServicesNodePorts:     resource.MustParse("0"),
			},
		},
	}

	// start up quota system
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)

	// old service was a load balancer, but updated version is a node port.
	existingService := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service", Namespace: "test", ResourceVersion: "1"},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}
	newService := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service", Namespace: "test"},
		Spec: api.ServiceSpec{
			Type:  api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{Port: 1234}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(newService, existingService, api.Kind("Service").WithVersion("version"), newService.Namespace, newService.Name, api.Resource("services").WithVersion("version"), "", admission.Update, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(kubeClient.Actions()) == 0 {
		t.Errorf("Expected a client action")
	}

	// the only action should have been to update the quota (since we should not have fetched the previous item)
	expectedActionSet := sets.NewString(
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}

	// verify usage decremented the loadbalancer, and incremented the nodeport, but kept the service the same.
	decimatedActions := removeListWatch(kubeClient.Actions())
	lastActionIndex := len(decimatedActions) - 1
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*api.ResourceQuota)
	expectedUsage := api.ResourceQuota{
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceServices:              resource.MustParse("10"),
				api.ResourceServicesLoadBalancers: resource.MustParse("10"),
				api.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: api.ResourceList{
				api.ResourceServices:              resource.MustParse("1"),
				api.ResourceServicesLoadBalancers: resource.MustParse("0"),
				api.ResourceServicesNodePorts:     resource.MustParse("1"),
			},
		},
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

// TestAdmitHandlesCreatingUpdates verifies that admit handles updates which behave as creates
func TestAdmitHandlesCreatingUpdates(t *testing.T) {
	// in this scenario, there is an existing service
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceServices:              resource.MustParse("10"),
				api.ResourceServicesLoadBalancers: resource.MustParse("10"),
				api.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: api.ResourceList{
				api.ResourceServices:              resource.MustParse("1"),
				api.ResourceServicesLoadBalancers: resource.MustParse("1"),
				api.ResourceServicesNodePorts:     resource.MustParse("0"),
			},
		},
	}

	// start up quota system
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)

	// old service didn't exist, so this update is actually a create
	oldService := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service", Namespace: "test", ResourceVersion: ""},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}
	newService := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "service", Namespace: "test"},
		Spec: api.ServiceSpec{
			Type:  api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{Port: 1234}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(newService, oldService, api.Kind("Service").WithVersion("version"), newService.Namespace, newService.Name, api.Resource("services").WithVersion("version"), "", admission.Update, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(kubeClient.Actions()) == 0 {
		t.Errorf("Expected a client action")
	}

	// the only action should have been to update the quota (since we should not have fetched the previous item)
	expectedActionSet := sets.NewString(
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}

	// verify that the "old" object was ignored for calculating the new usage
	decimatedActions := removeListWatch(kubeClient.Actions())
	lastActionIndex := len(decimatedActions) - 1
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*api.ResourceQuota)
	expectedUsage := api.ResourceQuota{
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceServices:              resource.MustParse("10"),
				api.ResourceServicesLoadBalancers: resource.MustParse("10"),
				api.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: api.ResourceList{
				api.ResourceServices:              resource.MustParse("2"),
				api.ResourceServicesLoadBalancers: resource.MustParse("1"),
				api.ResourceServicesNodePorts:     resource.MustParse("1"),
			},
		},
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

// TestAdmitExceedQuotaLimit verifies that if a pod exceeded allowed usage that its rejected during admission.
func TestAdmitExceedQuotaLimit(t *testing.T) {
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1"),
				api.ResourceMemory: resource.MustParse("50Gi"),
				api.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)
	newPod := validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected an error exceeding quota")
	}
}

// TestAdmitEnforceQuotaConstraints verifies that if a quota tracks a particular resource that that resource is
// specified on the pod.  In this case, we create a quota that tracks cpu request, memory request, and memory limit.
// We ensure that a pod that does not specify a memory limit that it fails in admission.
func TestAdmitEnforceQuotaConstraints(t *testing.T) {
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:          resource.MustParse("3"),
				api.ResourceMemory:       resource.MustParse("100Gi"),
				api.ResourceLimitsMemory: resource.MustParse("200Gi"),
				api.ResourcePods:         resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:          resource.MustParse("1"),
				api.ResourceMemory:       resource.MustParse("50Gi"),
				api.ResourceLimitsMemory: resource.MustParse("100Gi"),
				api.ResourcePods:         resource.MustParse("3"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)
	// verify all values are specified as required on the quota
	newPod := validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("200m", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected an error because the pod does not specify a memory limit")
	}
	// verify the requests and limits are actually valid (in this case, we fail because the limits < requests)
	newPod = validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("200m", "2Gi"), getResourceList("100m", "1Gi")))
	err = handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected an error because the pod does not specify a memory limit")
	}
}

// TestAdmitPodInNamespaceWithoutQuota ensures that if a namespace has no quota, that a pod can get in
func TestAdmitPodInNamespaceWithoutQuota(t *testing.T) {
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "other", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:          resource.MustParse("3"),
				api.ResourceMemory:       resource.MustParse("100Gi"),
				api.ResourceLimitsMemory: resource.MustParse("200Gi"),
				api.ResourcePods:         resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:          resource.MustParse("1"),
				api.ResourceMemory:       resource.MustParse("50Gi"),
				api.ResourceLimitsMemory: resource.MustParse("100Gi"),
				api.ResourcePods:         resource.MustParse("3"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	liveLookupCache, err := lru.New(100)
	if err != nil {
		t.Fatal(err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	quotaAccessor.liveLookupCache = liveLookupCache
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	// Add to the index
	indexer.Add(resourceQuota)
	newPod := validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("200m", "")))
	// Add to the lru cache so we do not do a live client lookup
	liveLookupCache.Add(newPod.Namespace, liveLookupEntry{expiry: time.Now().Add(time.Duration(30 * time.Second)), items: []*api.ResourceQuota{}})
	err = handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Did not expect an error because the pod is in a different namespace than the quota")
	}
}

// TestAdmitBelowTerminatingQuotaLimit ensures that terminating pods are charged to the right quota.
// It creates a terminating and non-terminating quota, and creates a terminating pod.
// It ensures that the terminating quota is incremented, and the non-terminating quota is not.
func TestAdmitBelowTerminatingQuotaLimit(t *testing.T) {
	resourceQuotaNonTerminating := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota-non-terminating", Namespace: "test", ResourceVersion: "124"},
		Spec: api.ResourceQuotaSpec{
			Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeNotTerminating},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1"),
				api.ResourceMemory: resource.MustParse("50Gi"),
				api.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	resourceQuotaTerminating := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota-terminating", Namespace: "test", ResourceVersion: "124"},
		Spec: api.ResourceQuotaSpec{
			Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeTerminating},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1"),
				api.ResourceMemory: resource.MustParse("50Gi"),
				api.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuotaTerminating, resourceQuotaNonTerminating)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuotaNonTerminating)
	indexer.Add(resourceQuotaTerminating)

	// create a pod that has an active deadline
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	activeDeadlineSeconds := int64(30)
	newPod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(kubeClient.Actions()) == 0 {
		t.Errorf("Expected a client action")
	}

	expectedActionSet := sets.NewString(
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}

	decimatedActions := removeListWatch(kubeClient.Actions())
	lastActionIndex := len(decimatedActions) - 1
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*api.ResourceQuota)

	// ensure only the quota-terminating was updated
	if usage.Name != resourceQuotaTerminating.Name {
		t.Errorf("Incremented the wrong quota, expected %v, actual %v", resourceQuotaTerminating.Name, usage.Name)
	}

	expectedUsage := api.ResourceQuota{
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("3"),
				api.ResourceMemory: resource.MustParse("100Gi"),
				api.ResourcePods:   resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1100m"),
				api.ResourceMemory: resource.MustParse("52Gi"),
				api.ResourcePods:   resource.MustParse("4"),
			},
		},
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

// TestAdmitBelowBestEffortQuotaLimit creates a best effort and non-best effort quota.
// It verifies that best effort pods are properly scoped to the best effort quota document.
func TestAdmitBelowBestEffortQuotaLimit(t *testing.T) {
	resourceQuotaBestEffort := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota-besteffort", Namespace: "test", ResourceVersion: "124"},
		Spec: api.ResourceQuotaSpec{
			Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeBestEffort},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourcePods: resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourcePods: resource.MustParse("3"),
			},
		},
	}
	resourceQuotaNotBestEffort := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota-not-besteffort", Namespace: "test", ResourceVersion: "124"},
		Spec: api.ResourceQuotaSpec{
			Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeNotBestEffort},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourcePods: resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourcePods: resource.MustParse("3"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuotaBestEffort, resourceQuotaNotBestEffort)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuotaBestEffort)
	indexer.Add(resourceQuotaNotBestEffort)

	// create a pod that is best effort because it does not make a request for anything
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	expectedActionSet := sets.NewString(
		strings.Join([]string{"update", "resourcequotas", "status"}, "-"),
	)
	actionSet := sets.NewString()
	for _, action := range kubeClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}
	decimatedActions := removeListWatch(kubeClient.Actions())
	lastActionIndex := len(decimatedActions) - 1
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*api.ResourceQuota)

	if usage.Name != resourceQuotaBestEffort.Name {
		t.Errorf("Incremented the wrong quota, expected %v, actual %v", resourceQuotaBestEffort.Name, usage.Name)
	}

	expectedUsage := api.ResourceQuota{
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourcePods: resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourcePods: resource.MustParse("4"),
			},
		},
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

func removeListWatch(in []testcore.Action) []testcore.Action {
	decimatedActions := []testcore.Action{}
	// list and watch resource quota is done to maintain our cache, so that's expected.  Remove them from results
	for i := range in {
		if in[i].Matches("list", "resourcequotas") || in[i].Matches("watch", "resourcequotas") {
			continue
		}

		decimatedActions = append(decimatedActions, in[i])
	}
	return decimatedActions
}

// TestAdmitBestEffortQuotaLimitIgnoresBurstable validates that a besteffort quota does not match a resource
// guaranteed pod.
func TestAdmitBestEffortQuotaLimitIgnoresBurstable(t *testing.T) {
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota-besteffort", Namespace: "test", ResourceVersion: "124"},
		Spec: api.ResourceQuotaSpec{
			Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeBestEffort},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourcePods: resource.MustParse("5"),
			},
			Used: api.ResourceList{
				api.ResourcePods: resource.MustParse("3"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	decimatedActions := removeListWatch(kubeClient.Actions())
	if len(decimatedActions) != 0 {
		t.Errorf("Expected no client actions because the incoming pod did not match best effort quota: %v", kubeClient.Actions())
	}
}

func TestHasUsageStats(t *testing.T) {
	testCases := map[string]struct {
		a        api.ResourceQuota
		expected bool
	}{
		"empty": {
			a:        api.ResourceQuota{Status: api.ResourceQuotaStatus{Hard: api.ResourceList{}}},
			expected: true,
		},
		"hard-only": {
			a: api.ResourceQuota{
				Status: api.ResourceQuotaStatus{
					Hard: api.ResourceList{
						api.ResourceMemory: resource.MustParse("1Gi"),
					},
					Used: api.ResourceList{},
				},
			},
			expected: false,
		},
		"hard-used": {
			a: api.ResourceQuota{
				Status: api.ResourceQuotaStatus{
					Hard: api.ResourceList{
						api.ResourceMemory: resource.MustParse("1Gi"),
					},
					Used: api.ResourceList{
						api.ResourceMemory: resource.MustParse("500Mi"),
					},
				},
			},
			expected: true,
		},
	}
	for testName, testCase := range testCases {
		if result := hasUsageStats(&testCase.a); result != testCase.expected {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, result)
		}
	}
}

// TestAdmissionSetsMissingNamespace verifies that if an object lacks a
// namespace, it will be set.
func TestAdmissionSetsMissingNamespace(t *testing.T) {
	namespace := "test"
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: namespace, ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU: resource.MustParse("3"),
			},
			Used: api.ResourceList{
				api.ResourceCPU: resource.MustParse("1"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})

	computeResources := []api.ResourceName{
		api.ResourcePods,
		api.ResourceCPU,
	}

	usageFunc := func(object runtime.Object) api.ResourceList {
		pod, ok := object.(*api.Pod)
		if !ok {
			t.Fatalf("Expected pod, got %T", object)
		}
		if pod.Namespace != namespace {
			t.Errorf("Expected pod with different namespace: %q != %q", pod.Namespace, namespace)
		}
		return core.PodUsageFunc(pod)
	}

	podEvaluator := &generic.GenericEvaluator{
		Name:              "Test-Evaluator.Pod",
		InternalGroupKind: api.Kind("Pod"),
		InternalOperationResources: map[admission.Operation][]api.ResourceName{
			admission.Create: computeResources,
		},
		ConstraintsFunc:      core.PodConstraintsFunc,
		MatchedResourceNames: computeResources,
		MatchesScopeFunc:     core.PodMatchesScopeFunc,
		UsageFunc:            usageFunc,
	}

	registry := &generic.GenericRegistry{
		InternalEvaluators: map[schema.GroupKind]quota.Evaluator{
			podEvaluator.GroupKind(): podEvaluator,
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)
	evaluator.(*quotaEvaluator).registry = registry

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)
	newPod := validPod("pod-without-namespace", 1, getResourceRequirements(getResourceList("1", "2Gi"), getResourceList("", "")))

	// unset the namespace
	newPod.ObjectMeta.Namespace = ""

	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Got unexpected error: %v", err)
	}
	if newPod.Namespace != namespace {
		t.Errorf("Got unexpected pod namespace: %q != %q", newPod.Namespace, namespace)
	}
}

// TestAdmitRejectsNegativeUsage verifies that usage for any measured resource cannot be negative.
func TestAdmitRejectsNegativeUsage(t *testing.T) {
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourcePersistentVolumeClaims: resource.MustParse("3"),
				api.ResourceRequestsStorage:        resource.MustParse("100Gi"),
			},
			Used: api.ResourceList{
				api.ResourcePersistentVolumeClaims: resource.MustParse("1"),
				api.ResourceRequestsStorage:        resource.MustParse("10Gi"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)
	// verify quota rejects negative pvc storage requests
	newPvc := validPersistentVolumeClaim("not-allowed-pvc", getResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("-1Gi")}, api.ResourceList{}))
	err := handler.Admit(admission.NewAttributesRecord(newPvc, nil, api.Kind("PersistentVolumeClaim").WithVersion("version"), newPvc.Namespace, newPvc.Name, api.Resource("persistentvolumeclaims").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected an error because the pvc has negative storage usage")
	}

	// verify quota accepts non-negative pvc storage requests
	newPvc = validPersistentVolumeClaim("not-allowed-pvc", getResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("1Gi")}, api.ResourceList{}))
	err = handler.Admit(admission.NewAttributesRecord(newPvc, nil, api.Kind("PersistentVolumeClaim").WithVersion("version"), newPvc.Namespace, newPvc.Name, api.Resource("persistentvolumeclaims").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

// TestAdmitWhenUnrelatedResourceExceedsQuota verifies that if resource X exceeds quota, it does not prohibit resource Y from admission.
func TestAdmitWhenUnrelatedResourceExceedsQuota(t *testing.T) {
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceServices: resource.MustParse("3"),
				api.ResourcePods:     resource.MustParse("4"),
			},
			Used: api.ResourceList{
				api.ResourceServices: resource.MustParse("4"),
				api.ResourcePods:     resource.MustParse("1"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	quotaAccessor, _ := newQuotaAccessor(kubeClient)
	quotaAccessor.indexer = indexer
	go quotaAccessor.Run(stopCh)
	evaluator := NewQuotaEvaluator(quotaAccessor, install.NewRegistry(nil, nil), nil, 5, stopCh)

	handler := &quotaAdmission{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		evaluator: evaluator,
	}
	indexer.Add(resourceQuota)

	// create a pod that should pass existing quota
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")))
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}
