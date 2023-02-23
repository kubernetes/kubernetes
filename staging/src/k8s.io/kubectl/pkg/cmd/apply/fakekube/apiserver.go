/*
Copyright 2023 The Kubernetes Authors.

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

package fakekube

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	memcached "k8s.io/client-go/discovery/cached/memory"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog/v2"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

// FakeKube implements (basic functionality of) a fake kube-apiserver, at the HTTP level.
// By implementing at the HTTP level, we are able to use all the existing clients,
// and don't have to mock them individually.
// It should only be used from tests (and requires a *testing.T)
type FakeKube struct {
	t         *testing.T
	resources map[schema.GroupVersionResource]*fakeKubeResource

	mutex   sync.Mutex
	nextUID int64

	discoveryClient discovery.CachedDiscoveryInterface
	restMapper      meta.RESTMapper
	restConfig      *restclient.Config

	beforeApplyHooks  []BeforeApplyHook
	beforeCreateHooks []BeforeCreateHook
	beforeDeleteHooks []BeforeDeleteHook
	beforeGetHooks    []BeforeGetHook
	beforeListHooks   []BeforeListHook
}

// BeforeApplyHook is the type for hook methods that should be called before an apply operation.
// If they return a non-nil response, processing will stop and that response will be returned.
type BeforeApplyHook func(req *ApplyRequest) *http.Response

// BeforeCreateHook is the type for hook methods that should be called before an apply operation.
// If they return a non-nil response, processing will stop and that response will be returned.
type BeforeCreateHook func(req *CreateRequest) *http.Response

// BeforeDeleteHook is the type for hook methods that should be called before a delete operation.
// If they return a non-nil response, processing will stop and that response will be returned.
type BeforeDeleteHook func(req *DeleteRequest) *http.Response

// BeforeGetHook is the type for hook methods that should be called before a get-object operation.
// If they return a non-nil response, processing will stop and that response will be returned.
type BeforeGetHook func(req *GetRequest) *http.Response

// BeforeListHook is the type for hook methods that should be called before a list-objects operation.
// If they return a non-nil response, processing will stop and that response will be returned.
type BeforeListHook func(req *ListRequest) *http.Response

// NewFakeKube is a constructor for a FakeKube fake kube-apiserver.
func NewFakeKube(t *testing.T) *FakeKube {
	k := &FakeKube{
		t:         t,
		resources: make(map[schema.GroupVersionResource]*fakeKubeResource),
		nextUID:   1,
	}

	k.addResource("Namespace", schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"})
	k.addResource("Secret", schema.GroupVersionResource{Group: "", Version: "v1", Resource: "secrets"})
	k.addResource("Service", schema.GroupVersionResource{Group: "", Version: "v1", Resource: "services"})
	k.addResource("ReplicationController", schema.GroupVersionResource{Group: "", Version: "v1", Resource: "replicationcontrollers"})

	restConfig := &restclient.Config{}
	restConfig.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
		return k
	}
	k.restConfig = restConfig

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(restConfig)
	if err != nil {
		t.Fatalf("error from NewDiscoveryClientForConfig: %v", err)
	}
	k.discoveryClient = memcached.NewMemCacheClient(discoveryClient)

	mapper := restmapper.NewDeferredDiscoveryRESTMapper(k.discoveryClient)
	expander := restmapper.NewShortcutExpander(mapper, discoveryClient)
	k.restMapper = expander

	return k
}

// AddBeforeApplyHook registers a hook to be run before every apply operation.
func (f *FakeKube) AddBeforeApplyHook(hook BeforeApplyHook) {
	f.beforeApplyHooks = append(f.beforeApplyHooks, hook)
}

// AddBeforeCreateHook registers a hook to be run before every create operation.
func (f *FakeKube) AddBeforeCreateHook(hook BeforeCreateHook) {
	f.beforeCreateHooks = append(f.beforeCreateHooks, hook)
}

// AddBeforeDeleteHook registers a hook to be run before every delete operation.
func (f *FakeKube) AddBeforeDeleteHook(hook BeforeDeleteHook) {
	f.beforeDeleteHooks = append(f.beforeDeleteHooks, hook)
}

// AddBeforeGetHook registers a hook to be run before every get-object operation.
func (f *FakeKube) AddBeforeGetHook(hook BeforeGetHook) {
	f.beforeGetHooks = append(f.beforeGetHooks, hook)
}

// AddBeforeListHook registers a hook to be run before every list-objects operation.
func (f *FakeKube) AddBeforeListHook(hook BeforeListHook) {
	f.beforeListHooks = append(f.beforeListHooks, hook)
}

// addResource registers the given type in the internal data structures.s
func (f *FakeKube) addResource(kind string, gvr schema.GroupVersionResource) {
	r := &fakeKubeResource{
		objects: make(map[types.NamespacedName]*unstructured.Unstructured),
	}
	r.Kind = kind
	r.ListKind = kind + "List"
	r.APIVersion = gvr.GroupVersion().Identifier()

	f.resources[gvr] = r
}

// fakeKubeResource tracks a known kubernetes API resource
type fakeKubeResource struct {
	APIVersion string
	Kind       string
	ListKind   string
	objects    map[types.NamespacedName]*unstructured.Unstructured
}

// RoundTrip implements http.RoundTripper, so we can be used without even listening on a port.
func (k *FakeKube) RoundTrip(req *http.Request) (*http.Response, error) {
	k.t.Logf("http request %v %s", req.Method, req.URL)

	tokens := strings.Split(strings.TrimPrefix(req.URL.Path, "/"), "/")

	if len(tokens) == 4 {
		if tokens[0] == "api" && tokens[1] == "v1" {
			id := types.NamespacedName{Namespace: "", Name: tokens[3]}
			gvr := schema.GroupVersionResource{Version: "v1", Resource: tokens[2]}

			if req.Method == http.MethodGet {
				op := &GetRequest{
					httpRequest: req,
					GVR:         gvr,
					ID:          id,
				}
				return k.doGet(op)
			}

			if req.Method == http.MethodDelete {
				op := &DeleteRequest{
					httpRequest: req,
					GVR:         gvr,
					ID:          id,
				}
				return k.doDelete(op)
			}
		}
	}

	if len(tokens) == 6 {
		if tokens[0] == "api" && tokens[1] == "v1" && tokens[2] == "namespaces" {
			id := types.NamespacedName{Namespace: tokens[3], Name: tokens[5]}
			gvr := schema.GroupVersionResource{Version: "v1", Resource: tokens[4]}

			if req.Method == http.MethodGet {
				op := &GetRequest{
					httpRequest: req,
					GVR:         gvr,
					ID:          id,
				}
				return k.doGet(op)
			}

			if req.Method == http.MethodDelete {
				op := &DeleteRequest{
					httpRequest: req,
					GVR:         gvr,
					ID:          id,
				}
				return k.doDelete(op)
			}

			if req.Method == http.MethodPatch {
				op := &ApplyRequest{
					httpRequest: req,
					GVR:         gvr,
					ID:          id,
				}
				return k.doApply(op)
			}
		}
	}

	if len(tokens) == 5 {
		if tokens[0] == "api" && tokens[1] == "v1" && tokens[2] == "namespaces" {
			gvr := schema.GroupVersionResource{Version: "v1", Resource: tokens[4]}

			if req.Method == http.MethodGet {
				op := &ListRequest{
					httpRequest: req,
					GVR:         gvr,
					Namespace:   tokens[3],
				}
				return k.doList(op)
			}
		}
	}

	if len(tokens) == 3 {
		if tokens[0] == "api" && tokens[1] == "v1" {
			gvr := schema.GroupVersionResource{Version: "v1", Resource: tokens[2]}

			if req.Method == http.MethodPost {
				op := &CreateRequest{
					httpRequest: req,
					GVR:         gvr,
				}
				return k.doCreate(op)
			}

			if req.Method == http.MethodGet {
				op := &ListRequest{
					httpRequest: req,
					GVR:         gvr,
				}
				return k.doList(op)
			}
		}
	}

	if len(tokens) == 2 {
		if tokens[0] == "api" && tokens[1] == "v1" {
			if req.Method == http.MethodGet {
				response := metav1.APIResourceList{}
				response.Kind = "APIResourceList"
				response.GroupVersion = "v1"
				response.APIResources = append(response.APIResources, metav1.APIResource{
					Name:       "namespaces",
					Namespaced: false,
					Kind:       "Namespace",
				})
				response.APIResources = append(response.APIResources, metav1.APIResource{
					Name:       "secrets",
					Namespaced: true,
					Kind:       "Secret",
				})
				response.APIResources = append(response.APIResources, metav1.APIResource{
					Name:       "services",
					Namespaced: true,
					Kind:       "Service",
				})
				response.APIResources = append(response.APIResources, metav1.APIResource{
					Name:       "replicationcontrollers",
					Namespaced: true,
					Kind:       "ReplicationController",
				})
				j, err := json.Marshal(response)
				if err != nil {
					return nil, fmt.Errorf("error building response: %w", err)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil
			}
		}
	}

	if len(tokens) == 2 {
		if tokens[0] == "openapi" && tokens[1] == "v2" {
			klog.Warningf("sending stub openapi/v2 response")
			response := []byte("")
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(response))}, nil
		}
		if tokens[0] == "openapi" && tokens[1] == "v3" {
			klog.Warningf("sending stub openapi/v3 response")
			response := []byte("")
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(response))}, nil
		}
	}

	if len(tokens) == 1 {
		if tokens[0] == "api" {
			if req.Method == http.MethodGet {
				response := metav1.APIVersions{}
				response.Kind = "APIVersions"
				response.Versions = []string{"v1"}
				j, err := json.Marshal(response)
				if err != nil {
					return nil, fmt.Errorf("error building response: %w", err)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil

			}
		}
		if tokens[0] == "apis" {
			if req.Method == http.MethodGet {
				response := metav1.APIGroupList{}
				response.Kind = "APIGroupList"
				response.APIVersion = "v1"
				// group := metav1.APIGroup{
				// 	Name:
				// }
				// response.Groups = append(response.Groups, group)

				j, err := json.Marshal(response)
				if err != nil {
					return nil, fmt.Errorf("error building response: %w", err)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil
			}
		}
	}

	return nil, fmt.Errorf("unexpected request in FakeKube: %v %v", req.Method, req.URL.Path)
}

// ApplyRequest is a parsed Apply request.
type ApplyRequest struct {
	httpRequest *http.Request

	GVR schema.GroupVersionResource
	ID  types.NamespacedName
}

// doApply services an apply-object request.
func (k *FakeKube) doApply(req *ApplyRequest) (*http.Response, error) {
	for _, hook := range k.beforeApplyHooks {
		response := hook(req)
		if response != nil {
			return response, nil
		}
	}

	body, err := io.ReadAll(req.httpRequest.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading request body: %w", err)
	}
	u := &unstructured.Unstructured{}
	if err := u.UnmarshalJSON(body); err != nil {
		return nil, fmt.Errorf("error parsing request body: %w", err)
	}

	id := types.NamespacedName{Namespace: u.GetNamespace(), Name: u.GetName()}
	if id != req.ID {
		klog.Warningf("id in apply object %v did not match id in URL %v", id, req.ID)
		return &http.Response{StatusCode: http.StatusBadRequest, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}
	resource := k.resources[req.GVR]
	if resource == nil {
		return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}

	existing := resource.objects[id]
	var returnObject *unstructured.Unstructured
	if existing == nil {
		// Treat as a create
		uid := k.generateUID()
		u.SetUID(uid)

		resource.objects[id] = u
		returnObject = u
	} else {
		// Merge values
		klog.Warningf("apply merging is stub-implemented")
		uid := existing.GetUID()
		u.DeepCopyInto(existing)
		existing.SetUID(uid)

		resource.objects[id] = existing
		returnObject = existing
	}

	j, err := returnObject.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("error building response: %w", err)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil
}

// CreateRequest is a parsed Create request.
type CreateRequest struct {
	httpRequest *http.Request

	GVR schema.GroupVersionResource
}

// doCreate services an Create-object request.
func (k *FakeKube) doCreate(req *CreateRequest) (*http.Response, error) {
	for _, hook := range k.beforeCreateHooks {
		response := hook(req)
		if response != nil {
			return response, nil
		}
	}

	body, err := io.ReadAll(req.httpRequest.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading request body: %w", err)
	}
	u := &unstructured.Unstructured{}
	if err := u.UnmarshalJSON(body); err != nil {
		return nil, fmt.Errorf("error parsing request body: %w", err)
	}

	uid := k.generateUID()
	u.SetUID(uid)

	id := types.NamespacedName{Namespace: u.GetNamespace(), Name: u.GetName()}
	resource := k.resources[req.GVR]
	if resource == nil {
		return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}
	resource.objects[id] = u

	j, err := u.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("error building response: %w", err)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil
}

// DeleteRequest is a parsed delete operation.
type DeleteRequest struct {
	httpRequest *http.Request

	GVR schema.GroupVersionResource
	ID  types.NamespacedName
}

// doDelete services a delete request.
func (k *FakeKube) doDelete(req *DeleteRequest) (*http.Response, error) {
	for _, hook := range k.beforeDeleteHooks {
		response := hook(req)
		if response != nil {
			return response, nil
		}
	}

	resource := k.resources[req.GVR]
	if resource == nil {
		return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}

	obj := resource.objects[req.ID]
	if obj == nil {
		return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}

	j, err := obj.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("error building response: %w", err)
	}

	delete(resource.objects, req.ID)
	return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil
}

// ReadObject reads the specified object directly, bypassing any hooks.
func (k *FakeKube) ReadObject(gvr schema.GroupVersionResource, namespace, name string) (*unstructured.Unstructured, error) {
	resource := k.resources[gvr]
	if resource == nil {
		return nil, fmt.Errorf("resource %v not found", gvr)
	}

	id := types.NamespacedName{Namespace: namespace, Name: name}
	obj := resource.objects[id]
	if obj == nil {
		return nil, fmt.Errorf("object not found")
	}

	obj = obj.DeepCopy()
	return obj, nil
}

// GetRequest is a parsed operation to read a single object.
type GetRequest struct {
	httpRequest *http.Request

	GVR schema.GroupVersionResource
	ID  types.NamespacedName
}

// doGet services a delete request.
func (k *FakeKube) doGet(req *GetRequest) (*http.Response, error) {
	for _, hook := range k.beforeGetHooks {
		response := hook(req)
		if response != nil {
			return response, nil
		}
	}

	resource := k.resources[req.GVR]
	if resource == nil {
		return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}

	obj := resource.objects[req.ID]
	if obj == nil {
		return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}

	j, err := obj.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("error building response: %w", err)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil
}

// ListRequest is a parsed operation to list some object.
type ListRequest struct {
	httpRequest *http.Request

	GVR       schema.GroupVersionResource
	Namespace string
}

// doList services a list objects request.
func (k *FakeKube) doList(req *ListRequest) (*http.Response, error) {
	for _, hook := range k.beforeListHooks {
		response := hook(req)
		if response != nil {
			return response, nil
		}
	}

	out := &unstructured.UnstructuredList{}

	resource := k.resources[req.GVR]
	if resource == nil {
		return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: nil}, nil
	}

	var labelSelector labels.Selector
	labelSelectorString := req.httpRequest.URL.Query().Get("labelSelector")
	if labelSelectorString != "" {
		ls, err := metav1.ParseToLabelSelector(labelSelectorString)
		if err != nil {
			return nil, fmt.Errorf("invalid labelSelector %q", labelSelectorString)
		}
		sel, err := metav1.LabelSelectorAsSelector(ls)
		if err != nil {
			return nil, fmt.Errorf("invalid labelSelector %q", labelSelectorString)
		}

		labelSelector = sel
	}
	for _, obj := range resource.objects {
		if req.Namespace != "" {
			if obj.GetNamespace() != req.Namespace {
				continue
			}
		}
		if labelSelector != nil {
			if !labelSelector.Matches(labels.Set(obj.GetLabels())) {
				continue
			}
		}
		out.Items = append(out.Items, *obj)
	}

	out.SetAPIVersion(resource.APIVersion)
	out.SetKind(resource.ListKind)

	j, err := out.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("error building response: %w", err)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader(j))}, nil
}

func (k *FakeKube) generateUID() types.UID {
	k.mutex.Lock()
	defer k.mutex.Unlock()

	n := k.nextUID
	k.nextUID++
	return types.UID(fmt.Sprintf("%v", n))
}

func (k *FakeKube) AsRESTClientGetter() genericclioptions.RESTClientGetter {
	return &restClientGetter{kube: k}
}

// restClientGetter is a minimal genericclioptions.RESTClientGetter, mapping to the fake kube-apiserver.
type restClientGetter struct {
	kube *FakeKube
}

// ToRESTConfig implements genericclioptions.RESTClientGetter
func (g *restClientGetter) ToRESTConfig() (*restclient.Config, error) {
	return g.kube.restConfig, nil
}

// ToDiscoveryClient implements genericclioptions.RESTClientGetter
func (g *restClientGetter) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	return g.kube.discoveryClient, nil
}

// ToRESTMapper implements genericclioptions.RESTClientGetter
func (g *restClientGetter) ToRESTMapper() (meta.RESTMapper, error) {
	return g.kube.restMapper, nil
}

// ToRawKubeConfigLoader implements genericclioptions.RESTClientGetter
func (g *restClientGetter) ToRawKubeConfigLoader() clientcmd.ClientConfig {
	return &clientConfig{
		kube:             g.kube,
		defaultNamespace: "test",
	}
}

// clientConfig is a minimal clientcmd.ClientConfig, mapping to the fake kube-apiserver.
type clientConfig struct {
	kube             *FakeKube
	defaultNamespace string
}

// RawConfig implements clientcmd.ClientConfig
func (c *clientConfig) RawConfig() (clientcmdapi.Config, error) {
	panic("not implemented")
}

// ClientConfig implements clientcmd.ClientConfig
func (c *clientConfig) ClientConfig() (*restclient.Config, error) {
	panic("not implemented")
}

// Namespace implements clientcmd.ClientConfig
func (c *clientConfig) Namespace() (string, bool, error) {
	return c.defaultNamespace, true, nil
}

// ConfigAccess implements clientcmd.ClientConfig
func (c *clientConfig) ConfigAccess() clientcmd.ConfigAccess {
	panic("not implemented")
}
