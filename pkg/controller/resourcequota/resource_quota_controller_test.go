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
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
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

type testRESTMapper struct {
	meta.RESTMapper
}

func (_ *testRESTMapper) Reset() {}

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

type fakeServerResources struct {
	PreferredResources []*metav1.APIResourceList
	Error              error
	Lock               sync.Mutex
	InterfaceUsedCount int
}

func (_ *fakeServerResources) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	return nil, nil
}

func (_ *fakeServerResources) ServerResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

func (f *fakeServerResources) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.InterfaceUsedCount++
	return f.PreferredResources, f.Error
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

func (_ *fakeServerResources) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

// testServerAndClientConfig returns a server that listens and a config that can reference it
func testServerAndClientConfig(handler func(http.ResponseWriter, *http.Request)) (*httptest.Server, *restclient.Config) {
	srv := httptest.NewServer(http.HandlerFunc(handler))
	config := &restclient.Config{
		Host: srv.URL,
	}
	return srv, config
}

func expectSyncNotBlocked(fakeDiscoveryClient *fakeServerResources) error {
	before := fakeDiscoveryClient.getInterfaceUsedCount()
	t := 1 * time.Second
	time.Sleep(t)
	after := fakeDiscoveryClient.getInterfaceUsedCount()
	if before == after {
		return fmt.Errorf("discoveryClient.ServerPreferredResources() called %d times over %v", after-before, t)
	}
	return nil
}

type quotaController struct {
	*ResourceQuotaController
	stop chan struct{}
}

func setupQuotaController(t *testing.T, kubeClient kubernetes.Interface, lister quota.ListerForResourceFunc) quotaController {
	config := &restclient.Config{}
	tweakableRM := meta.NewDefaultRESTMapper(nil)
	rm := &testRESTMapper{meta.MultiRESTMapper{tweakableRM, testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme)}}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	twoResources := map[schema.GroupVersionResource]struct{}{
		{Version: "v1", Resource: "pods"}:                          {},
		{Group: "example.com", Version: "v1", Resource: "foobars"}: {},
	}

	sharedInformerFactory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	quotaConfiguration := install.NewQuotaConfigurationForControllers(lister)
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)

	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		QuotaClient:               kubeClient.CoreV1(),
		ResourceQuotaInformer:     sharedInformerFactory.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		IgnoredResourcesFunc:      quotaConfiguration.IgnoredResources,
		DiscoveryFunc:             mockDiscoveryFunc,
		Registry:                  generic.NewRegistry(quotaConfiguration.Evaluators()),
		InformersStarted:          alwaysStarted,
		RESTMapper:                rm,
		DynamicClient:             dynamicClient,
		QuotableResources:         twoResources,
		SharedInformerFactory:     sharedInformerFactory,
	}
	qc, err := NewResourceQuotaController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	go sharedInformerFactory.Start(stop)
	return quotaController{qc, stop}
}

func newTestUnstructured() []runtime.Object {
	return []runtime.Object{
		&unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "example.com/v1",
				"kind":       "foobar",
				"content": map[string]interface{}{
					"key": "value",
				},
			},
		},
	}
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
	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "foobars"}
	listersForResourceConfig := map[schema.GroupVersionResource]cache.GenericLister{
		gvr: newGenericLister(gvr.GroupResource(), newTestUnstructured()),
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
			name:             "status, missing usage",
			expectedPriority: true,
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

func TestQuotaControllerConstruction(t *testing.T) {
	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "foobar"}
	listersForResourceConfig := map[schema.GroupVersionResource]cache.GenericLister{
		gvr: newGenericLister(gvr.GroupResource(), newTestUnstructured()),
	}

	config := &restclient.Config{}
	tweakableRM := meta.NewDefaultRESTMapper(nil)
	rm := &testRESTMapper{meta.MultiRESTMapper{tweakableRM, testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme)}}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	podResource := map[schema.GroupVersionResource]struct{}{
		{Version: "v1", Resource: "pods"}: {},
	}
	twoResources := map[schema.GroupVersionResource]struct{}{
		{Version: "v1", Resource: "pods"}:                         {},
		{Group: "example.com", Version: "v1", Resource: "foobar"}: {},
	}

	client := fake.NewSimpleClientset()
	sharedInformers := informers.NewSharedInformerFactory(client, 0)
	quotaConfiguration := install.NewQuotaConfigurationForControllers(mockListerForResourceFunc(listersForResourceConfig))
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)
	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		QuotaClient:               client.CoreV1(),
		ResourceQuotaInformer:     sharedInformers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		IgnoredResourcesFunc:      quotaConfiguration.IgnoredResources,
		DiscoveryFunc:             mockDiscoveryFunc,
		Registry:                  generic.NewRegistry(quotaConfiguration.Evaluators()),
		InformersStarted:          alwaysStarted,
		DynamicClient:             dynamicClient,
		RESTMapper:                rm,
		QuotableResources:         twoResources,
		SharedInformerFactory:     sharedInformers,
	}
	qc, err := NewResourceQuotaController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, 2, len(qc.quotaMonitor.monitors))

	// Make sure resource monitor syncing creates and stops resource monitors.
	err = qc.resyncMonitors(podResource)
	if err != nil {
		t.Errorf("Failed removing a monitor: %v", err)
	}
	assert.Equal(t, 1, len(qc.quotaMonitor.monitors))

	// Make sure the syncing mechanism also works after Run() has been called
	stopCh := make(chan struct{})
	defer close(stopCh)
	go qc.Run(1, stopCh)

	err = qc.resyncMonitors(twoResources)
	if err != nil {
		t.Errorf("Failed adding a monitor: %v", err)
	}
	assert.Equal(t, 2, len(qc.quotaMonitor.monitors))

	err = qc.resyncMonitors(podResource)
	if err != nil {
		t.Errorf("Failed removing a monitor: %v", err)
	}
	assert.Equal(t, 1, len(qc.quotaMonitor.monitors))
}

// TestRQListWatcher tests that the list and watch functions correctly convert ListOptions
func TestRQListWatcher(t *testing.T) {
	testHandler := &fakeActionHandler{}
	srv, clientConfig := testServerAndClientConfig(testHandler.ServeHTTP)
	defer srv.Close()

	podResource := schema.GroupVersionResource{Version: "v1", Resource: "pods"}
	dynamicClient, err := dynamic.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	lw := listWatcher(dynamicClient, podResource)
	lw.DisableChunking = true
	if _, err := lw.Watch(metav1.ListOptions{ResourceVersion: "1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := lw.List(metav1.ListOptions{ResourceVersion: "1"}); err != nil {
		t.Fatal(err)
	}
	if e, a := 2, len(testHandler.actions); e != a {
		t.Errorf("expect %d requests, got %d", e, a)
	}
	if e, a := "resourceVersion=1&watch=true", testHandler.actions[0].query; e != a {
		t.Errorf("expect %s, got %s", e, a)
	}
	if e, a := "resourceVersion=1", testHandler.actions[1].query; e != a {
		t.Errorf("expect %s, got %s", e, a)
	}
}

// TestGetQuotableResources ensures GetQuotableResources always returns
// something usable regardless of discovery output.
func TestGetQuotableResources(t *testing.T) {
	tests := map[string]struct {
		serverResources   []*metav1.APIResourceList
		err               error
		quotableResources map[schema.GroupVersionResource]struct{}
	}{
		"no error": {
			serverResources: []*metav1.APIResourceList{
				{
					// Valid GroupVersion
					GroupVersion: "apps/v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: metav1.Verbs{"create", "list", "delete"}},
						{Name: "services", Namespaced: true, Kind: "Service"},
					},
				},
				{
					// Invalid GroupVersion, should be ignored
					GroupVersion: "foo//whatever",
					APIResources: []metav1.APIResource{
						{Name: "bars", Namespaced: true, Kind: "Bar", Verbs: metav1.Verbs{"create", "list", "delete"}},
					},
				},
				{
					// Valid GroupVersion, missing required verbs, should be ignored
					GroupVersion: "acme/v1",
					APIResources: []metav1.APIResource{
						{Name: "widgets", Namespaced: true, Kind: "Widget", Verbs: metav1.Verbs{"delete"}},
					},
				},
			},
			err: nil,
			quotableResources: map[schema.GroupVersionResource]struct{}{
				{Group: "apps", Version: "v1", Resource: "pods"}: {},
			},
		},
		"nonspecific failure, includes usable results": {
			serverResources: []*metav1.APIResourceList{
				{
					GroupVersion: "apps/v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: metav1.Verbs{"create", "list", "delete"}},
						{Name: "services", Namespaced: true, Kind: "Service"},
					},
				},
			},
			err: fmt.Errorf("internal error"),
			quotableResources: map[schema.GroupVersionResource]struct{}{
				{Group: "apps", Version: "v1", Resource: "pods"}: {},
			},
		},
		"partial discovery failure, includes usable results": {
			serverResources: []*metav1.APIResourceList{
				{
					GroupVersion: "apps/v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: metav1.Verbs{"create", "list", "delete"}},
						{Name: "services", Namespaced: true, Kind: "Service"},
					},
				},
			},
			err: &discovery.ErrGroupDiscoveryFailed{
				Groups: map[schema.GroupVersion]error{
					{Group: "foo", Version: "v1"}: fmt.Errorf("discovery failure"),
				},
			},
			quotableResources: map[schema.GroupVersionResource]struct{}{
				{Group: "apps", Version: "v1", Resource: "pods"}: {},
			},
		},
		"discovery failure, no results": {
			serverResources:   nil,
			err:               fmt.Errorf("internal error"),
			quotableResources: map[schema.GroupVersionResource]struct{}{},
		},
	}

	for name, test := range tests {
		t.Logf("testing %q", name)
		client := &fakeServerResources{
			PreferredResources: test.serverResources,
			Error:              test.err,
		}
		actual := GetQuotableResources(client)
		if !reflect.DeepEqual(test.quotableResources, actual) {
			t.Errorf("expected resources:\n%v\ngot:\n%v", test.quotableResources, actual)
		}
	}
}

// TestQuotaControllerSync ensures that a discovery client error
// will not cause the quota controller to block infinitely.

func TestQuotaControllerSync(t *testing.T) {
	serverResources := []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: metav1.Verbs{"delete", "list", "watch"}},
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
		},
	}
	srv, clientConfig := testServerAndClientConfig(testHandler.ServeHTTP)
	defer srv.Close()
	clientConfig.ContentConfig.NegotiatedSerializer = nil
	client, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	rm := &testRESTMapper{testrestmapper.TestOnlyStaticRESTMapper(legacyscheme.Scheme)}
	dynamicClient, err := dynamic.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	podResource := map[schema.GroupVersionResource]struct{}{
		{Group: "", Version: "v1", Resource: "pods"}: {},
	}

	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "foobar"}
	listersForResourceConfig := map[schema.GroupVersionResource]cache.GenericLister{
		gvr: newGenericLister(gvr.GroupResource(), newTestUnstructured()),
	}
	quotaConfiguration := install.NewQuotaConfigurationForControllers(mockListerForResourceFunc(listersForResourceConfig))
	sharedInformers := informers.NewSharedInformerFactory(client, 0)

	alwaysStarted := make(chan struct{})
	close(alwaysStarted)

	resourceQuotaControllerOptions := &ResourceQuotaControllerOptions{
		QuotaClient:               client.CoreV1(),
		ResourceQuotaInformer:     sharedInformers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		IgnoredResourcesFunc:      quotaConfiguration.IgnoredResources,
		DiscoveryFunc:             mockDiscoveryFunc,
		Registry:                  generic.NewRegistry(quotaConfiguration.Evaluators()),
		InformersStarted:          alwaysStarted,
		DynamicClient:             dynamicClient,
		RESTMapper:                rm,
		QuotableResources:         podResource,
		SharedInformerFactory:     sharedInformers,
	}
	qc, err := NewResourceQuotaController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatal(err)
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	go qc.Run(1, stopCh)
	go qc.Sync(fakeDiscoveryClient, 10*time.Millisecond, stopCh)

	// Wait until the sync discovers the initial resources
	fmt.Printf("Test output")
	time.Sleep(1 * time.Second)

	err = expectSyncNotBlocked(fakeDiscoveryClient)
	if err != nil {
		t.Fatalf("Expected quotacontroller.Sync to be running but it is blocked: %v", err)
	}

	// Simulate the discovery client returning an error
	fakeDiscoveryClient.setPreferredResources(nil)
	fakeDiscoveryClient.setError(fmt.Errorf("Error calling discoveryClient.ServerPreferredResources()"))

	// Wait until sync discovers the change
	time.Sleep(1 * time.Second)

	// Remove the error from being returned and see if the quota controller sync is still working
	fakeDiscoveryClient.setPreferredResources(serverResources)
	fakeDiscoveryClient.setError(nil)

	err = expectSyncNotBlocked(fakeDiscoveryClient)
	if err != nil {
		t.Fatalf("Expected quotacontroller.Sync to still be running but it is blocked: %v", err)
	}
}
