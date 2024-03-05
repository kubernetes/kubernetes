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
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/resourcequota"
	resourcequotaapi "k8s.io/apiserver/pkg/admission/plugin/resourcequota/apis/resourcequota"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/quota/v1/install"
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

func getVolumeResourceRequirements(requests, limits api.ResourceList) api.VolumeResourceRequirements {
	res := api.VolumeResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func validPod(name string, numContainers int, resources api.ResourceRequirements) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test"},
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

func validPodWithPriority(name string, numContainers int, resources api.ResourceRequirements, priorityClass string) *api.Pod {
	pod := validPod(name, numContainers, resources)
	if priorityClass != "" {
		pod.Spec.PriorityClassName = priorityClass
	}
	return pod
}

func validPersistentVolumeClaim(name string, resources api.VolumeResourceRequirements) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: resources,
		},
	}
}

func createHandler(kubeClient kubernetes.Interface, informerFactory informers.SharedInformerFactory, stopCh chan struct{}) (*resourcequota.QuotaAdmission, error) {
	return createHandlerWithConfig(kubeClient, informerFactory, nil, stopCh)
}

func createHandlerWithConfig(kubeClient kubernetes.Interface, informerFactory informers.SharedInformerFactory, config *resourcequotaapi.Configuration, stopCh chan struct{}) (*resourcequota.QuotaAdmission, error) {
	if config == nil {
		config = &resourcequotaapi.Configuration{}
	}
	quotaConfiguration := install.NewQuotaConfigurationForAdmission()

	handler, err := resourcequota.NewResourceQuota(config, 5)
	if err != nil {
		return nil, err
	}

	initializers := admission.PluginInitializers{
		genericadmissioninitializer.New(kubeClient, nil, informerFactory, nil, nil, stopCh),
		kubeapiserveradmission.NewPluginInitializer(nil, nil, quotaConfiguration, nil),
	}
	initializers.Initialize(handler)

	return handler, admission.ValidateInitialization(handler)
}

// TestAdmissionIgnoresDelete verifies that the admission controller ignores delete operations
func TestAdmissionIgnoresDelete(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	namespace := "default"
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(nil, nil, api.Kind("Pod").WithVersion("version"), namespace, "name", corev1.Resource("pods").WithVersion("version"), "", admission.Delete, &metav1.DeleteOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("ResourceQuota should admit all deletes: %v", err)
	}
}

// TestAdmissionIgnoresSubresources verifies that the admission controller ignores subresources
// It verifies that creation of a pod that would have exceeded quota is properly failed
// It verifies that create operations to a subresource that would have exceeded quota would succeed
func TestAdmissionIgnoresSubresources(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{}
	resourceQuota.Name = "quota"
	resourceQuota.Namespace = "test"
	resourceQuota.Status = corev1.ResourceQuotaStatus{
		Hard: corev1.ResourceList{},
		Used: corev1.ResourceList{},
	}
	resourceQuota.Status.Hard[corev1.ResourceMemory] = resource.MustParse("2Gi")
	resourceQuota.Status.Used[corev1.ResourceMemory] = resource.MustParse("1Gi")
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	newPod := validPod("123", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Errorf("Expected an error because the pod exceeded allowed quota")
	}
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "subresource", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Did not expect an error because the action went to a subresource: %v", err)
	}
}

// TestAdmitBelowQuotaLimit verifies that a pod when created has its usage reflected on the quota
func TestAdmitBelowQuotaLimit(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:   resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("50Gi"),
				corev1.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
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
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*corev1.ResourceQuota)
	expectedUsage := corev1.ResourceQuota{
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:   resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1100m"),
				corev1.ResourceMemory: resource.MustParse("52Gi"),
				corev1.ResourcePods:   resource.MustParse("4"),
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

// TestAdmitDryRun verifies that a pod when created with dry-run doesn not have its usage reflected on the quota
// and that dry-run requests can still be rejected if they would exceed the quota
func TestAdmitDryRun(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:   resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("50Gi"),
				corev1.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)

	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, true, nil), nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	newPod = validPod("too-large-pod", 1, getResourceRequirements(getResourceList("100m", "60Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, true, nil), nil)
	if err == nil {
		t.Errorf("Expected error but got none")
	}

	if len(kubeClient.Actions()) != 0 {
		t.Errorf("Expected no client action on dry-run")
	}
}

// TestAdmitHandlesOldObjects verifies that admit handles updates correctly with old objects
func TestAdmitHandlesOldObjects(t *testing.T) {
	// in this scenario, the old quota was based on a service type=loadbalancer
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("10"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("10"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("1"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("1"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("0"),
			},
		},
	}

	// start up quota system
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)

	// old service was a load balancer, but updated version is a node port.
	existingService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test", ResourceVersion: "1"},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}
	newService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test"},
		Spec: api.ServiceSpec{
			Type:  api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{Port: 1234}},
		},
	}
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newService, existingService, api.Kind("Service").WithVersion("version"), newService.Namespace, newService.Name, corev1.Resource("services").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil), nil)
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
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*corev1.ResourceQuota)

	// Verify service usage. Since we don't add negative values, the corev1.ResourceServicesLoadBalancers
	// will remain on last reported value
	expectedUsage := corev1.ResourceQuota{
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("10"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("10"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("1"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("1"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("1"),
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

func TestAdmitHandlesNegativePVCUpdates(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("3"),
				corev1.ResourceRequestsStorage:        resource.MustParse("100Gi"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("1"),
				corev1.ResourceRequestsStorage:        resource.MustParse("10Gi"),
			},
		},
	}

	// start up quota system
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)

	oldPVC := &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-to-update", Namespace: "test", ResourceVersion: "1"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: getVolumeResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("10Gi")}, api.ResourceList{}),
		},
	}

	newPVC := &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-to-update", Namespace: "test"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: getVolumeResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("5Gi")}, api.ResourceList{}),
		},
	}

	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPVC, oldPVC, api.Kind("PersistentVolumeClaim").WithVersion("version"), newPVC.Namespace, newPVC.Name, corev1.Resource("persistentvolumeclaims").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(kubeClient.Actions()) != 0 {
		t.Errorf("No client action should be taken in case of negative updates")
	}
}

func TestAdmitHandlesPVCUpdates(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("3"),
				corev1.ResourceRequestsStorage:        resource.MustParse("100Gi"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("1"),
				corev1.ResourceRequestsStorage:        resource.MustParse("10Gi"),
			},
		},
	}

	// start up quota system
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)

	oldPVC := &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-to-update", Namespace: "test", ResourceVersion: "1"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: getVolumeResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("10Gi")}, api.ResourceList{}),
		},
	}

	newPVC := &api.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-to-update", Namespace: "test"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: getVolumeResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("15Gi")}, api.ResourceList{}),
		},
	}

	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPVC, oldPVC, api.Kind("PersistentVolumeClaim").WithVersion("version"), newPVC.Namespace, newPVC.Name, corev1.Resource("persistentvolumeclaims").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil), nil)
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

	decimatedActions := removeListWatch(kubeClient.Actions())
	lastActionIndex := len(decimatedActions) - 1
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*corev1.ResourceQuota)
	expectedUsage := corev1.ResourceQuota{
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("3"),
				corev1.ResourceRequestsStorage:        resource.MustParse("100Gi"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("1"),
				corev1.ResourceRequestsStorage:        resource.MustParse("15Gi"),
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
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("10"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("10"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("1"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("1"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("0"),
			},
		},
	}

	// start up quota system
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)

	// old service didn't exist, so this update is actually a create
	oldService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test", ResourceVersion: ""},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}
	newService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test"},
		Spec: api.ServiceSpec{
			Type:  api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{Port: 1234}},
		},
	}
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newService, oldService, api.Kind("Service").WithVersion("version"), newService.Namespace, newService.Name, corev1.Resource("services").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil), nil)
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
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*corev1.ResourceQuota)
	expectedUsage := corev1.ResourceQuota{
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("10"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("10"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("10"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceServices:              resource.MustParse("2"),
				corev1.ResourceServicesLoadBalancers: resource.MustParse("1"),
				corev1.ResourceServicesNodePorts:     resource.MustParse("1"),
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
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:   resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("50Gi"),
				corev1.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	newPod := validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Errorf("Expected an error exceeding quota")
	}
}

// TestAdmitEnforceQuotaConstraints verifies that if a quota tracks a particular resource that that resource is
// specified on the pod.  In this case, we create a quota that tracks cpu request, memory request, and memory limit.
// We ensure that a pod that does not specify a memory limit that it fails in admission.
func TestAdmitEnforceQuotaConstraints(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:          resource.MustParse("3"),
				corev1.ResourceMemory:       resource.MustParse("100Gi"),
				corev1.ResourceLimitsMemory: resource.MustParse("200Gi"),
				corev1.ResourcePods:         resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:          resource.MustParse("1"),
				corev1.ResourceMemory:       resource.MustParse("50Gi"),
				corev1.ResourceLimitsMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:         resource.MustParse("3"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	// verify all values are specified as required on the quota
	newPod := validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("200m", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Errorf("Expected an error because the pod does not specify a memory limit")
	}
}

// TestAdmitPodInNamespaceWithoutQuota ensures that if a namespace has no quota, that a pod can get in
func TestAdmitPodInNamespaceWithoutQuota(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "other", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:          resource.MustParse("3"),
				corev1.ResourceMemory:       resource.MustParse("100Gi"),
				corev1.ResourceLimitsMemory: resource.MustParse("200Gi"),
				corev1.ResourcePods:         resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:          resource.MustParse("1"),
				corev1.ResourceMemory:       resource.MustParse("50Gi"),
				corev1.ResourceLimitsMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:         resource.MustParse("3"),
			},
		},
	}

	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	// Add to the index
	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("200m", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Did not expect an error because the pod is in a different namespace than the quota: %v", err)
	}
}

// TestAdmitBelowTerminatingQuotaLimit ensures that terminating pods are charged to the right quota.
// It creates a terminating and non-terminating quota, and creates a terminating pod.
// It ensures that the terminating quota is incremented, and the non-terminating quota is not.
func TestAdmitBelowTerminatingQuotaLimit(t *testing.T) {
	resourceQuotaNonTerminating := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota-non-terminating", Namespace: "test", ResourceVersion: "124"},
		Spec: corev1.ResourceQuotaSpec{
			Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeNotTerminating},
		},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:   resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("50Gi"),
				corev1.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	resourceQuotaTerminating := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota-terminating", Namespace: "test", ResourceVersion: "124"},
		Spec: corev1.ResourceQuotaSpec{
			Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeTerminating},
		},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:   resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("50Gi"),
				corev1.ResourcePods:   resource.MustParse("3"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuotaTerminating, resourceQuotaNonTerminating)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuotaNonTerminating)
	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuotaTerminating)

	// create a pod that has an active deadline
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", "")))
	activeDeadlineSeconds := int64(30)
	newPod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
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
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*corev1.ResourceQuota)

	// ensure only the quota-terminating was updated
	if usage.Name != resourceQuotaTerminating.Name {
		t.Errorf("Incremented the wrong quota, expected %v, actual %v", resourceQuotaTerminating.Name, usage.Name)
	}

	expectedUsage := corev1.ResourceQuota{
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("3"),
				corev1.ResourceMemory: resource.MustParse("100Gi"),
				corev1.ResourcePods:   resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1100m"),
				corev1.ResourceMemory: resource.MustParse("52Gi"),
				corev1.ResourcePods:   resource.MustParse("4"),
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
	resourceQuotaBestEffort := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota-besteffort", Namespace: "test", ResourceVersion: "124"},
		Spec: corev1.ResourceQuotaSpec{
			Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeBestEffort},
		},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("3"),
			},
		},
	}
	resourceQuotaNotBestEffort := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota-not-besteffort", Namespace: "test", ResourceVersion: "124"},
		Spec: corev1.ResourceQuotaSpec{
			Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeNotBestEffort},
		},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("3"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuotaBestEffort, resourceQuotaNotBestEffort)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuotaBestEffort)
	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuotaNotBestEffort)

	// create a pod that is best effort because it does not make a request for anything
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
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
	usage := decimatedActions[lastActionIndex].(testcore.UpdateAction).GetObject().(*corev1.ResourceQuota)

	if usage.Name != resourceQuotaBestEffort.Name {
		t.Errorf("Incremented the wrong quota, expected %v, actual %v", resourceQuotaBestEffort.Name, usage.Name)
	}

	expectedUsage := corev1.ResourceQuota{
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("4"),
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
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota-besteffort", Namespace: "test", ResourceVersion: "124"},
		Spec: corev1.ResourceQuotaSpec{
			Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeBestEffort},
		},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("5"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("3"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	decimatedActions := removeListWatch(kubeClient.Actions())
	if len(decimatedActions) != 0 {
		t.Errorf("Expected no client actions because the incoming pod did not match best effort quota: %v", kubeClient.Actions())
	}
}

// TestAdmissionSetsMissingNamespace verifies that if an object lacks a
// namespace, it will be set.
func TestAdmissionSetsMissingNamespace(t *testing.T) {
	namespace := "test"
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: namespace, ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("3"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePods: resource.MustParse("1"),
			},
		},
	}

	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	newPod := validPod("pod-without-namespace", 1, getResourceRequirements(getResourceList("1", "2Gi"), getResourceList("", "")))

	// unset the namespace
	newPod.ObjectMeta.Namespace = ""

	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Got unexpected error: %v", err)
	}
	if newPod.Namespace != namespace {
		t.Errorf("Got unexpected pod namespace: %q != %q", newPod.Namespace, namespace)
	}
}

// TestAdmitRejectsNegativeUsage verifies that usage for any measured resource cannot be negative.
func TestAdmitRejectsNegativeUsage(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("3"),
				corev1.ResourceRequestsStorage:        resource.MustParse("100Gi"),
			},
			Used: corev1.ResourceList{
				corev1.ResourcePersistentVolumeClaims: resource.MustParse("1"),
				corev1.ResourceRequestsStorage:        resource.MustParse("10Gi"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)
	// verify quota rejects negative pvc storage requests
	newPvc := validPersistentVolumeClaim("not-allowed-pvc", getVolumeResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("-1Gi")}, api.ResourceList{}))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPvc, nil, api.Kind("PersistentVolumeClaim").WithVersion("version"), newPvc.Namespace, newPvc.Name, corev1.Resource("persistentvolumeclaims").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Errorf("Expected an error because the pvc has negative storage usage")
	}

	// verify quota accepts non-negative pvc storage requests
	newPvc = validPersistentVolumeClaim("not-allowed-pvc", getVolumeResourceRequirements(api.ResourceList{api.ResourceStorage: resource.MustParse("1Gi")}, api.ResourceList{}))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPvc, nil, api.Kind("PersistentVolumeClaim").WithVersion("version"), newPvc.Namespace, newPvc.Name, corev1.Resource("persistentvolumeclaims").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

// TestAdmitWhenUnrelatedResourceExceedsQuota verifies that if resource X exceeds quota, it does not prohibit resource Y from admission.
func TestAdmitWhenUnrelatedResourceExceedsQuota(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceServices: resource.MustParse("3"),
				corev1.ResourcePods:     resource.MustParse("4"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceServices: resource.MustParse("4"),
				corev1.ResourcePods:     resource.MustParse("1"),
			},
		},
	}
	stopCh := make(chan struct{})
	defer close(stopCh)

	kubeClient := fake.NewSimpleClientset(resourceQuota)
	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	handler, err := createHandler(kubeClient, informerFactory, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	informerFactory.Core().V1().ResourceQuotas().Informer().GetIndexer().Add(resourceQuota)

	// create a pod that should pass existing quota
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

// TestAdmitLimitedResourceNoQuota verifies if a limited resource is configured with no quota, it cannot be consumed.
func TestAdmitLimitedResourceNoQuota(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable consumption of cpu unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "pods",
				MatchContains: []string{"cpu"},
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	newPod := validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Errorf("Expected an error for consuming a limited resource without quota.")
	}
}

// TestAdmitLimitedResourceNoQuotaIgnoresNonMatchingResources shows it ignores non matching resources in config.
func TestAdmitLimitedResourceNoQuotaIgnoresNonMatchingResources(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable consumption of cpu unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "services",
				MatchContains: []string{"services"},
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

// TestAdmitLimitedResourceWithQuota verifies if a limited resource is configured with quota, it can be consumed.
func TestAdmitLimitedResourceWithQuota(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("10"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("1"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable consumption of cpu unless there is a covering quota.
	// disable consumption of cpu unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "pods",
				MatchContains: []string{"requests.cpu"}, // match on "requests.cpu" only
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	indexer.Add(resourceQuota)
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestAdmitLimitedResourceWithMultipleQuota verifies if a limited resource is configured with quota, it can be consumed if one matches.
func TestAdmitLimitedResourceWithMultipleQuota(t *testing.T) {
	resourceQuota1 := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota1", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("10"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("1"),
			},
		},
	}
	resourceQuota2 := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota2", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("10Gi"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota1, resourceQuota2)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable consumption of cpu unless there is a covering quota.
	// disable consumption of cpu unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "pods",
				MatchContains: []string{"requests.cpu"}, // match on "requests.cpu" only
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	indexer.Add(resourceQuota1)
	indexer.Add(resourceQuota2)
	newPod := validPod("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestAdmitLimitedResourceWithQuotaThatDoesNotCover verifies if a limited resource is configured the quota must cover the resource.
func TestAdmitLimitedResourceWithQuotaThatDoesNotCover(t *testing.T) {
	resourceQuota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("10Gi"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
	}
	kubeClient := fake.NewSimpleClientset(resourceQuota)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable consumption of cpu unless there is a covering quota.
	// disable consumption of cpu unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "pods",
				MatchContains: []string{"cpu"}, // match on "cpu" only
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	indexer.Add(resourceQuota)
	newPod := validPod("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")))
	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Fatalf("Expected an error since the quota did not cover cpu")
	}
}

// TestAdmitLimitedScopeWithQuota verifies if a limited scope is configured the quota must cover the resource.
func TestAdmitLimitedScopeWithCoverQuota(t *testing.T) {
	testCases := []struct {
		description  string
		testPod      *api.Pod
		quota        *corev1.ResourceQuota
		anotherQuota *corev1.ResourceQuota
		config       *resourcequotaapi.Configuration
		expErr       string
	}{
		{
			description: "Covering quota exists for configured limited scope PriorityClassNameExists.",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "fake-priority"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
				},
			},
			expErr: "",
		},
		{
			description: "configured limited scope PriorityClassNameExists and limited cpu resource. No covering quota for cpu and pod admit fails.",
			testPod:     validPodWithPriority("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "fake-priority"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
						MatchContains: []string{"requests.cpu"}, // match on "requests.cpu" only
					},
				},
			},
			expErr: "insufficient quota to consume: requests.cpu",
		},
		{
			description: "Covering quota does not exist for configured limited scope PriorityClassNameExists.",
			testPod:     validPodWithPriority("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "fake-priority"),
			quota:       &corev1.ResourceQuota{},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{PriorityClass Exists []}]",
		},
		{
			description: "Covering quota does not exist for configured limited scope resourceQuotaBestEffort",
			testPod:     validPodWithPriority("not-allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")), "fake-priority"),
			quota:       &corev1.ResourceQuota{},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{BestEffort Exists []}]",
		},
		{
			description: "Covering quota exist for configured limited scope resourceQuotaBestEffort",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")), "fake-priority"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota-besteffort", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeBestEffort},
				},
				Status: corev1.ResourceQuotaStatus{
					Hard: corev1.ResourceList{
						corev1.ResourcePods: resource.MustParse("5"),
					},
					Used: corev1.ResourceList{
						corev1.ResourcePods: resource.MustParse("3"),
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
				},
			},
			expErr: "",
		},
		{
			description: "Two scopes,BestEffort and PriorityClassIN, in two LimitedResources. Neither matches pod. Pod allowed",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", "")), "fake-priority"),
			quota:       &corev1.ResourceQuota{},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "",
		},
		{
			description: "Two scopes,BestEffort and PriorityClassIN, in two LimitedResources. Only BestEffort scope matches pod. Pod admit fails because covering quota is missing for BestEffort scope",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")), "fake-priority"),
			quota:       &corev1.ResourceQuota{},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{BestEffort Exists []}]",
		},
		{
			description: "Two scopes,BestEffort and PriorityClassIN, in two LimitedResources. Only PriorityClass scope matches pod. Pod admit fails because covering quota is missing for PriorityClass scope",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("100m", "1Gi"), getResourceList("", "")), "cluster-services"),
			quota:       &corev1.ResourceQuota{},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{PriorityClass In [cluster-services]}]",
		},
		{
			description: "Two scopes,BestEffort and PriorityClassIN, in two LimitedResources. Both the scopes matches pod. Pod admit fails because covering quota is missing for PriorityClass scope and BestEffort scope",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")), "cluster-services"),
			quota:       &corev1.ResourceQuota{},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{BestEffort Exists []} {PriorityClass In [cluster-services]}]",
		},
		{
			description: "Two scopes,BestEffort and PriorityClassIN, in two LimitedResources. Both the scopes matches pod. Quota available only for BestEffort scope. Pod admit fails because covering quota is missing for PriorityClass scope",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")), "cluster-services"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota-besteffort", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeBestEffort},
				},
				Status: corev1.ResourceQuotaStatus{
					Hard: corev1.ResourceList{
						corev1.ResourcePods: resource.MustParse("5"),
					},
					Used: corev1.ResourceList{
						corev1.ResourcePods: resource.MustParse("3"),
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{PriorityClass In [cluster-services]}]",
		},
		{
			description: "Two scopes,BestEffort and PriorityClassIN, in two LimitedResources. Both the scopes matches pod. Quota available only for PriorityClass scope. Pod admit fails because covering quota is missing for BestEffort scope",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")), "cluster-services"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{BestEffort Exists []}]",
		},
		{
			description: "Two scopes,BestEffort and PriorityClassIN, in two LimitedResources. Both the scopes matches pod. Quota available only for both the scopes. Pod admit success. No Error",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "")), "cluster-services"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota-besteffort", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					Scopes: []corev1.ResourceQuotaScope{corev1.ResourceQuotaScopeBestEffort},
				},
				Status: corev1.ResourceQuotaStatus{
					Hard: corev1.ResourceList{
						corev1.ResourcePods: resource.MustParse("5"),
					},
					Used: corev1.ResourceList{
						corev1.ResourcePods: resource.MustParse("3"),
					},
				},
			},
			anotherQuota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopeBestEffort,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "",
		},
		{
			description: "Pod allowed with priorityclass if limited scope PriorityClassNameExists not configured.",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "fake-priority"),
			quota:       &corev1.ResourceQuota{},
			config:      &resourcequotaapi.Configuration{},
			expErr:      "",
		},
		{
			description: "quota fails, though covering quota for configured limited scope PriorityClassNameExists exists.",
			testPod:     validPodWithPriority("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "20Gi"), getResourceList("", "")), "fake-priority"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists},
						},
					},
				},
				Status: corev1.ResourceQuotaStatus{
					Hard: corev1.ResourceList{
						corev1.ResourceMemory: resource.MustParse("10Gi"),
					},
					Used: corev1.ResourceList{
						corev1.ResourceMemory: resource.MustParse("1Gi"),
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists,
							},
						},
					},
				},
			},
			expErr: "forbidden: exceeded quota: quota, requested: memory=20Gi, used: memory=1Gi, limited: memory=10Gi",
		},
		{
			description: "Pod has different priorityclass than configured limited. Covering quota exists for configured limited scope PriorityClassIn.",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "fake-priority"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			expErr: "",
		},
		{
			description: "Pod has limited priorityclass. Covering quota exists for configured limited scope PriorityClassIn.",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "cluster-services"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"cluster-services"},
							},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"another-priorityclass-name", "cluster-services"},
							},
						},
					},
				},
			},
			expErr: "",
		},
		{
			description: "Pod has limited priorityclass. Covering quota  does not exist for configured limited scope PriorityClassIn.",
			testPod:     validPodWithPriority("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "cluster-services"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"another-priorityclass-name"},
							},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"another-priorityclass-name", "cluster-services"},
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{PriorityClass In [another-priorityclass-name cluster-services]}]",
		},
		{
			description: "From the above test case, just changing pod priority from cluster-services to another-priorityclass-name. expecting no error",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "another-priorityclass-name"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"another-priorityclass-name"},
							},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"another-priorityclass-name", "cluster-services"},
							},
						},
					},
				},
			},
			expErr: "",
		},
		{
			description: "Pod has limited priorityclass. Covering quota does NOT exists for configured limited scope PriorityClassIn.",
			testPod:     validPodWithPriority("not-allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "cluster-services"),
			quota:       &corev1.ResourceQuota{},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"another-priorityclass-name", "cluster-services"},
							},
						},
					},
				},
			},
			expErr: "insufficient quota to match these scopes: [{PriorityClass In [another-priorityclass-name cluster-services]}]",
		},
		{
			description: "Pod has limited priorityclass. Covering quota exists for configured limited scope PriorityClassIn through PriorityClassNameExists",
			testPod:     validPodWithPriority("allowed-pod", 1, getResourceRequirements(getResourceList("3", "2Gi"), getResourceList("", "")), "cluster-services"),
			quota: &corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "quota", Namespace: "test", ResourceVersion: "124"},
				Spec: corev1.ResourceQuotaSpec{
					ScopeSelector: &corev1.ScopeSelector{
						MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpExists},
						},
					},
				},
			},
			config: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource: "pods",
						MatchScopes: []corev1.ScopedResourceSelectorRequirement{
							{
								ScopeName: corev1.ResourceQuotaScopePriorityClass,
								Operator:  corev1.ScopeSelectorOpIn,
								Values:    []string{"another-priorityclass-name", "cluster-services"},
							},
						},
					},
				},
			},
			expErr: "",
		},
	}

	for _, testCase := range testCases {
		newPod := testCase.testPod
		config := testCase.config
		resourceQuota := testCase.quota
		kubeClient := fake.NewSimpleClientset(resourceQuota)
		if testCase.anotherQuota != nil {
			kubeClient = fake.NewSimpleClientset(resourceQuota, testCase.anotherQuota)
		}
		indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
		stopCh := make(chan struct{})
		defer close(stopCh)

		informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

		handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
		if err != nil {
			t.Errorf("Error occurred while creating admission plugin: %v", err)
		}

		indexer.Add(resourceQuota)
		if testCase.anotherQuota != nil {
			indexer.Add(testCase.anotherQuota)
		}
		err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, corev1.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
		if testCase.expErr == "" {
			if err != nil {
				t.Fatalf("Testcase, %v, failed with unexpected error: %v. ExpErr: %v", testCase.description, err, testCase.expErr)
			}
		} else {
			if !strings.Contains(fmt.Sprintf("%v", err), testCase.expErr) {
				t.Fatalf("Testcase, %v, failed with unexpected error: %v. ExpErr: %v", testCase.description, err, testCase.expErr)
			}
		}

	}
}

// TestAdmitZeroDeltaUsageWithoutCoveringQuota verifies that resource quota is not required for zero delta requests.
func TestAdmitZeroDeltaUsageWithoutCoveringQuota(t *testing.T) {

	kubeClient := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable services unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "services",
				MatchContains: []string{"services.loadbalancers"},
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	existingService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test", ResourceVersion: "1"},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}
	newService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test"},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}

	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newService, existingService, api.Kind("Service").WithVersion("version"), newService.Namespace, newService.Name, corev1.Resource("services").WithVersion("version"), "", admission.Update, &metav1.CreateOptions{}, false, nil), nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestAdmitRejectIncreaseUsageWithoutCoveringQuota verifies that resource quota is required for delta requests that increase usage.
func TestAdmitRejectIncreaseUsageWithoutCoveringQuota(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable services unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "services",
				MatchContains: []string{"services.loadbalancers"},
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	existingService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test", ResourceVersion: "1"},
		Spec: api.ServiceSpec{
			Type:  api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{Port: 1234}},
		},
	}
	newService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test"},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}

	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newService, existingService, api.Kind("Service").WithVersion("version"), newService.Namespace, newService.Name, corev1.Resource("services").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil), nil)
	if err == nil {
		t.Errorf("Expected an error for consuming a limited resource without quota.")
	}
}

// TestAdmitAllowDecreaseUsageWithoutCoveringQuota verifies that resource quota is not required for delta requests that decrease usage.
func TestAdmitAllowDecreaseUsageWithoutCoveringQuota(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)

	// disable services unless there is a covering quota.
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "services",
				MatchContains: []string{"services.loadbalancers"},
			},
		},
	}

	handler, err := createHandlerWithConfig(kubeClient, informerFactory, config, stopCh)
	if err != nil {
		t.Errorf("Error occurred while creating admission plugin: %v", err)
	}

	existingService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test", ResourceVersion: "1"},
		Spec:       api.ServiceSpec{Type: api.ServiceTypeLoadBalancer},
	}
	newService := &api.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "service", Namespace: "test"},
		Spec: api.ServiceSpec{
			Type:  api.ServiceTypeNodePort,
			Ports: []api.ServicePort{{Port: 1234}},
		},
	}

	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(newService, existingService, api.Kind("Service").WithVersion("version"), newService.Namespace, newService.Name, corev1.Resource("services").WithVersion("version"), "", admission.Update, &metav1.UpdateOptions{}, false, nil), nil)

	if err != nil {
		t.Errorf("Expected no error for decreasing a limited resource without quota, got %v", err)
	}
}
