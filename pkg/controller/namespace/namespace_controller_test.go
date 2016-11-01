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

package namespace

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"sync"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestFinalized(t *testing.T) {
	testNamespace := &api.Namespace{
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"a", "b"},
		},
	}
	if finalized(testNamespace) {
		t.Errorf("Unexpected result, namespace is not finalized")
	}
	testNamespace.Spec.Finalizers = []api.FinalizerName{}
	if !finalized(testNamespace) {
		t.Errorf("Expected object to be finalized")
	}
}

func TestFinalizeNamespaceFunc(t *testing.T) {
	mockClient := &fake.Clientset{}
	testNamespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            "test",
			ResourceVersion: "1",
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"kubernetes", "other"},
		},
	}
	finalizeNamespace(mockClient, testNamespace, api.FinalizerKubernetes)
	actions := mockClient.Actions()
	if len(actions) != 1 {
		t.Errorf("Expected 1 mock client action, but got %v", len(actions))
	}
	if !actions[0].Matches("create", "namespaces") || actions[0].GetSubresource() != "finalize" {
		t.Errorf("Expected finalize-namespace action %v", actions[0])
	}
	finalizers := actions[0].(core.CreateAction).GetObject().(*api.Namespace).Spec.Finalizers
	if len(finalizers) != 1 {
		t.Errorf("There should be a single finalizer remaining")
	}
	if "other" != string(finalizers[0]) {
		t.Errorf("Unexpected finalizer value, %v", finalizers[0])
	}
}

func testSyncNamespaceThatIsTerminating(t *testing.T, versions *unversioned.APIVersions) {
	now := unversioned.Now()
	namespaceName := "test"
	testNamespacePendingFinalize := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:              namespaceName,
			ResourceVersion:   "1",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"kubernetes"},
		},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceTerminating,
		},
	}
	testNamespaceFinalizeComplete := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:              namespaceName,
			ResourceVersion:   "1",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceTerminating,
		},
	}

	// when doing a delete all of content, we will do a GET of a collection, and DELETE of a collection by default
	dynamicClientActionSet := sets.NewString()
	groupVersionResources := testGroupVersionResources()
	for _, groupVersionResource := range groupVersionResources {
		urlPath := path.Join([]string{
			dynamic.LegacyAPIPathResolverFunc(unversioned.GroupVersionKind{Group: groupVersionResource.Group, Version: groupVersionResource.Version}),
			groupVersionResource.Group,
			groupVersionResource.Version,
			"namespaces",
			namespaceName,
			groupVersionResource.Resource,
		}...)
		dynamicClientActionSet.Insert((&fakeAction{method: "GET", path: urlPath}).String())
		dynamicClientActionSet.Insert((&fakeAction{method: "DELETE", path: urlPath}).String())
	}

	scenarios := map[string]struct {
		testNamespace          *api.Namespace
		kubeClientActionSet    sets.String
		dynamicClientActionSet sets.String
		gvrError               error
	}{
		"pending-finalize": {
			testNamespace: testNamespacePendingFinalize,
			kubeClientActionSet: sets.NewString(
				strings.Join([]string{"get", "namespaces", ""}, "-"),
				strings.Join([]string{"create", "namespaces", "finalize"}, "-"),
				strings.Join([]string{"list", "pods", ""}, "-"),
				strings.Join([]string{"delete", "namespaces", ""}, "-"),
			),
			dynamicClientActionSet: dynamicClientActionSet,
		},
		"complete-finalize": {
			testNamespace: testNamespaceFinalizeComplete,
			kubeClientActionSet: sets.NewString(
				strings.Join([]string{"get", "namespaces", ""}, "-"),
				strings.Join([]string{"delete", "namespaces", ""}, "-"),
			),
			dynamicClientActionSet: sets.NewString(),
		},
		"groupVersionResourceErr": {
			testNamespace: testNamespaceFinalizeComplete,
			kubeClientActionSet: sets.NewString(
				strings.Join([]string{"get", "namespaces", ""}, "-"),
				strings.Join([]string{"delete", "namespaces", ""}, "-"),
			),
			dynamicClientActionSet: sets.NewString(),
			gvrError:               fmt.Errorf("test error"),
		},
	}

	for scenario, testInput := range scenarios {
		testHandler := &fakeActionHandler{statusCode: 200}
		srv, clientConfig := testServerAndClientConfig(testHandler.ServeHTTP)
		defer srv.Close()

		mockClient := fake.NewSimpleClientset(testInput.testNamespace)
		clientPool := dynamic.NewClientPool(clientConfig, registered.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)

		fn := func() ([]unversioned.GroupVersionResource, error) {
			return groupVersionResources, nil
		}

		err := syncNamespace(mockClient, clientPool, &operationNotSupportedCache{m: make(map[operationKey]bool)}, fn, testInput.testNamespace, api.FinalizerKubernetes)
		if err != nil {
			t.Errorf("scenario %s - Unexpected error when synching namespace %v", scenario, err)
		}

		// validate traffic from kube client
		actionSet := sets.NewString()
		for _, action := range mockClient.Actions() {
			actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource().Resource, action.GetSubresource()}, "-"))
		}
		if !actionSet.Equal(testInput.kubeClientActionSet) {
			t.Errorf("scenario %s - mock client expected actions:\n%v\n but got:\n%v\nDifference:\n%v", scenario,
				testInput.kubeClientActionSet, actionSet, testInput.kubeClientActionSet.Difference(actionSet))
		}

		// validate traffic from dynamic client
		actionSet = sets.NewString()
		for _, action := range testHandler.actions {
			actionSet.Insert(action.String())
		}
		if !actionSet.Equal(testInput.dynamicClientActionSet) {
			t.Errorf("scenario %s - dynamic client expected actions:\n%v\n but got:\n%v\nDifference:\n%v", scenario,
				testInput.dynamicClientActionSet, actionSet, testInput.dynamicClientActionSet.Difference(actionSet))
		}
	}
}

func TestRetryOnConflictError(t *testing.T) {
	mockClient := &fake.Clientset{}
	numTries := 0
	retryOnce := func(kubeClient clientset.Interface, namespace *api.Namespace) (*api.Namespace, error) {
		numTries++
		if numTries <= 1 {
			return namespace, errors.NewConflict(api.Resource("namespaces"), namespace.Name, fmt.Errorf("ERROR!"))
		}
		return namespace, nil
	}
	namespace := &api.Namespace{}
	_, err := retryOnConflictError(mockClient, namespace, retryOnce)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if numTries != 2 {
		t.Errorf("Expected %v, but got %v", 2, numTries)
	}
}

func TestSyncNamespaceThatIsTerminatingNonExperimental(t *testing.T) {
	testSyncNamespaceThatIsTerminating(t, &unversioned.APIVersions{})
}

func TestSyncNamespaceThatIsTerminatingV1Beta1(t *testing.T) {
	testSyncNamespaceThatIsTerminating(t, &unversioned.APIVersions{Versions: []string{"extensions/v1beta1"}})
}

func TestSyncNamespaceThatIsActive(t *testing.T) {
	mockClient := &fake.Clientset{}
	testNamespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            "test",
			ResourceVersion: "1",
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"kubernetes"},
		},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceActive,
		},
	}
	fn := func() ([]unversioned.GroupVersionResource, error) {
		return testGroupVersionResources(), nil
	}
	err := syncNamespace(mockClient, nil, &operationNotSupportedCache{m: make(map[operationKey]bool)}, fn, testNamespace, api.FinalizerKubernetes)
	if err != nil {
		t.Errorf("Unexpected error when synching namespace %v", err)
	}
	if len(mockClient.Actions()) != 0 {
		t.Errorf("Expected no action from controller, but got: %v", mockClient.Actions())
	}
}

// testServerAndClientConfig returns a server that listens and a config that can reference it
func testServerAndClientConfig(handler func(http.ResponseWriter, *http.Request)) (*httptest.Server, *restclient.Config) {
	srv := httptest.NewServer(http.HandlerFunc(handler))
	config := &restclient.Config{
		Host: srv.URL,
	}
	return srv, config
}

// fakeAction records information about requests to aid in testing.
type fakeAction struct {
	method string
	path   string
}

// String returns method=path to aid in testing
func (f *fakeAction) String() string {
	return strings.Join([]string{f.method, f.path}, "=")
}

// fakeActionHandler holds a list of fakeActions received
type fakeActionHandler struct {
	// statusCode returned by this handler
	statusCode int

	lock    sync.Mutex
	actions []fakeAction
}

// ServeHTTP logs the action that occurred and always returns the associated status code
func (f *fakeActionHandler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.actions = append(f.actions, fakeAction{method: request.Method, path: request.URL.Path})
	response.Header().Set("Content-Type", runtime.ContentTypeJSON)
	response.WriteHeader(f.statusCode)
	response.Write([]byte("{\"kind\": \"List\",\"items\":null}"))
}

// testGroupVersionResources returns a mocked up set of resources across different api groups for testing namespace controller.
func testGroupVersionResources() []unversioned.GroupVersionResource {
	results := []unversioned.GroupVersionResource{}
	results = append(results, unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"})
	results = append(results, unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "services"})
	results = append(results, unversioned.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"})
	return results
}
