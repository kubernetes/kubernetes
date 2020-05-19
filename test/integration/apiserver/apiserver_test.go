/*
Copyright 2017 The Kubernetes Authors.

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

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"path"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/pager"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration/framework"
)

func setup(t *testing.T, groupVersions ...schema.GroupVersion) (*httptest.Server, clientset.Interface, framework.CloseFunc) {
	return setupWithResources(t, groupVersions, nil)
}

func setupWithOptions(t *testing.T, opts *framework.MasterConfigOptions, groupVersions ...schema.GroupVersion) (*httptest.Server, clientset.Interface, framework.CloseFunc) {
	return setupWithResourcesWithOptions(t, opts, groupVersions, nil)
}

func setupWithResources(t *testing.T, groupVersions []schema.GroupVersion, resources []schema.GroupVersionResource) (*httptest.Server, clientset.Interface, framework.CloseFunc) {
	return setupWithResourcesWithOptions(t, &framework.MasterConfigOptions{}, groupVersions, resources)
}

func setupWithResourcesWithOptions(t *testing.T, opts *framework.MasterConfigOptions, groupVersions []schema.GroupVersion, resources []schema.GroupVersionResource) (*httptest.Server, clientset.Interface, framework.CloseFunc) {
	masterConfig := framework.NewIntegrationTestMasterConfigWithOptions(opts)
	if len(groupVersions) > 0 || len(resources) > 0 {
		resourceConfig := master.DefaultAPIResourceConfigSource()
		resourceConfig.EnableVersions(groupVersions...)
		resourceConfig.EnableResources(resources...)
		masterConfig.ExtraConfig.APIResourceConfigSource = resourceConfig
	}
	masterConfig.GenericConfig.OpenAPIConfig = framework.DefaultOpenAPIConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	clientSet, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	return s, clientSet, closeFn
}

func verifyStatusCode(t *testing.T, verb, URL, body string, expectedStatusCode int) {
	// We dont use the typed Go client to send this request to be able to verify the response status code.
	bodyBytes := bytes.NewReader([]byte(body))
	req, err := http.NewRequest(verb, URL, bodyBytes)
	if err != nil {
		t.Fatalf("unexpected error: %v in sending req with verb: %s, URL: %s and body: %s", err, verb, URL, body)
	}
	transport := http.DefaultTransport
	klog.Infof("Sending request: %v", req)
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v in req: %v", err, req)
	}
	defer resp.Body.Close()
	b, _ := ioutil.ReadAll(resp.Body)
	if resp.StatusCode != expectedStatusCode {
		t.Errorf("Expected status %v, but got %v", expectedStatusCode, resp.StatusCode)
		t.Errorf("Body: %v", string(b))
	}
}

func newRS(namespace string) *apps.ReplicaSet {
	return &apps.ReplicaSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicaSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    namespace,
			GenerateName: "apiserver-test",
		},
		Spec: apps.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"name": "test"}},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": "test"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
				},
			},
		},
	}
}

var cascDel = `
{
  "kind": "DeleteOptions",
  "apiVersion": "v1",
  "orphanDependents": false
}
`

func Test4xxStatusCodeInvalidPatch(t *testing.T) {
	_, client, closeFn := setup(t)
	defer closeFn()

	obj := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
                },
		"spec": {
			"selector": {
				"matchLabels": {
					 "app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "nginx:latest"
					}]
				}
			}
		}
	}`)

	resp, err := client.CoreV1().RESTClient().Post().
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Body(obj).Do(context.TODO()).Get()
	if err != nil {
		t.Fatalf("Failed to create object: %v: %v", err, resp)
	}
	result := client.CoreV1().RESTClient().Patch(types.MergePatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Body([]byte(`{"metadata":{"annotations":{"foo":["bar"]}}}`)).Do(context.TODO())
	var statusCode int
	result.StatusCode(&statusCode)
	if statusCode != 422 {
		t.Fatalf("Expected status code to be 422, got %v (%#v)", statusCode, result)
	}
	result = client.CoreV1().RESTClient().Patch(types.StrategicMergePatchType).
		AbsPath("/apis/apps/v1").
		Namespace("default").
		Resource("deployments").
		Name("deployment").
		Body([]byte(`{"metadata":{"annotations":{"foo":["bar"]}}}`)).Do(context.TODO())
	result.StatusCode(&statusCode)
	if statusCode != 422 {
		t.Fatalf("Expected status code to be 422, got %v (%#v)", statusCode, result)
	}
}

func TestCacheControl(t *testing.T) {
	masterConfig := framework.NewIntegrationTestMasterConfigWithOptions(&framework.MasterConfigOptions{})
	masterConfig.GenericConfig.OpenAPIConfig = framework.DefaultOpenAPIConfig()
	master, _, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	rt, err := restclient.TransportFor(master.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	paths := []string{
		// untyped
		"/",
		// health
		"/healthz",
		// openapi
		"/openapi/v2",
		// discovery
		"/api",
		"/api/v1",
		"/apis",
		"/apis/apps",
		"/apis/apps/v1",
		// apis
		"/api/v1/namespaces",
		"/apis/apps/v1/deployments",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			req, err := http.NewRequest("GET", master.GenericAPIServer.LoopbackClientConfig.Host+path, nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := rt.RoundTrip(req)
			if err != nil {
				t.Fatal(err)
			}
			cc := resp.Header.Get("Cache-Control")
			if !strings.Contains(cc, "private") {
				t.Errorf("expected private cache-control, got %q", cc)
			}
		})
	}
}

// Tests that the apiserver returns 202 status code as expected.
func Test202StatusCode(t *testing.T) {
	s, clientSet, closeFn := setup(t)
	defer closeFn()

	ns := framework.CreateTestingNamespace("status-code", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	rsClient := clientSet.AppsV1().ReplicaSets(ns.Name)

	// 1. Create the resource without any finalizer and then delete it without setting DeleteOptions.
	// Verify that server returns 200 in this case.
	rs, err := rsClient.Create(context.TODO(), newRS(ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create rs: %v", err)
	}
	verifyStatusCode(t, "DELETE", s.URL+path.Join("/apis/apps/v1/namespaces", ns.Name, "replicasets", rs.Name), "", 200)

	// 2. Create the resource with a finalizer so that the resource is not immediately deleted and then delete it without setting DeleteOptions.
	// Verify that the apiserver still returns 200 since DeleteOptions.OrphanDependents is not set.
	rs = newRS(ns.Name)
	rs.ObjectMeta.Finalizers = []string{"kube.io/dummy-finalizer"}
	rs, err = rsClient.Create(context.TODO(), rs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create rs: %v", err)
	}
	verifyStatusCode(t, "DELETE", s.URL+path.Join("/apis/apps/v1/namespaces", ns.Name, "replicasets", rs.Name), "", 200)

	// 3. Create the resource and then delete it with DeleteOptions.OrphanDependents=false.
	// Verify that the server still returns 200 since the resource is immediately deleted.
	rs = newRS(ns.Name)
	rs, err = rsClient.Create(context.TODO(), rs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create rs: %v", err)
	}
	verifyStatusCode(t, "DELETE", s.URL+path.Join("/apis/apps/v1/namespaces", ns.Name, "replicasets", rs.Name), cascDel, 200)

	// 4. Create the resource with a finalizer so that the resource is not immediately deleted and then delete it with DeleteOptions.OrphanDependents=false.
	// Verify that the server returns 202 in this case.
	rs = newRS(ns.Name)
	rs.ObjectMeta.Finalizers = []string{"kube.io/dummy-finalizer"}
	rs, err = rsClient.Create(context.TODO(), rs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create rs: %v", err)
	}
	verifyStatusCode(t, "DELETE", s.URL+path.Join("/apis/apps/v1/namespaces", ns.Name, "replicasets", rs.Name), cascDel, 202)
}

func TestListResourceVersion0(t *testing.T) {
	var testcases = []struct {
		name              string
		watchCacheEnabled bool
	}{
		{
			name:              "watchCacheOn",
			watchCacheEnabled: true,
		},
		{
			name:              "watchCacheOff",
			watchCacheEnabled: false,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIListChunking, true)()
			etcdOptions := framework.DefaultEtcdOptions()
			etcdOptions.EnableWatchCache = tc.watchCacheEnabled
			s, clientSet, closeFn := setupWithOptions(t, &framework.MasterConfigOptions{EtcdOptions: etcdOptions})
			defer closeFn()

			ns := framework.CreateTestingNamespace("list-paging", s, t)
			defer framework.DeleteTestingNamespace(ns, s, t)

			rsClient := clientSet.AppsV1().ReplicaSets(ns.Name)

			for i := 0; i < 10; i++ {
				rs := newRS(ns.Name)
				rs.Name = fmt.Sprintf("test-%d", i)
				if _, err := rsClient.Create(context.TODO(), rs, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
			}

			pagerFn := func(opts metav1.ListOptions) (runtime.Object, error) {
				return rsClient.List(context.TODO(), opts)
			}

			p := pager.New(pager.SimplePageFunc(pagerFn))
			p.PageSize = 3
			listObj, _, err := p.List(context.Background(), metav1.ListOptions{ResourceVersion: "0"})
			if err != nil {
				t.Fatalf("Unexpected list error: %v", err)
			}
			items, err := meta.ExtractList(listObj)
			if err != nil {
				t.Fatalf("Failed to extract list from %v", listObj)
			}
			if len(items) != 10 {
				t.Errorf("Expected list size of 10 but got %d", len(items))
			}
		})
	}
}

func TestAPIListChunking(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIListChunking, true)()
	s, clientSet, closeFn := setup(t)
	defer closeFn()

	ns := framework.CreateTestingNamespace("list-paging", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	rsClient := clientSet.AppsV1().ReplicaSets(ns.Name)

	for i := 0; i < 4; i++ {
		rs := newRS(ns.Name)
		rs.Name = fmt.Sprintf("test-%d", i)
		if _, err := rsClient.Create(context.TODO(), rs, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	calls := 0
	firstRV := ""
	p := &pager.ListPager{
		PageSize: 1,
		PageFn: pager.SimplePageFunc(func(opts metav1.ListOptions) (runtime.Object, error) {
			calls++
			list, err := rsClient.List(context.TODO(), opts)
			if err != nil {
				return nil, err
			}
			if calls == 1 {
				firstRV = list.ResourceVersion
			}
			if calls == 2 {
				rs := newRS(ns.Name)
				rs.Name = "test-5"
				if _, err := rsClient.Create(context.TODO(), rs, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
			}
			return list, err
		}),
	}
	listObj, _, err := p.List(context.Background(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 4 {
		t.Errorf("unexpected list invocations: %d", calls)
	}
	list := listObj.(metav1.ListInterface)
	if len(list.GetContinue()) != 0 {
		t.Errorf("unexpected continue: %s", list.GetContinue())
	}
	if list.GetResourceVersion() != firstRV {
		t.Errorf("unexpected resource version: %s instead of %s", list.GetResourceVersion(), firstRV)
	}
	var names []string
	if err := meta.EachListItem(listObj, func(obj runtime.Object) error {
		rs := obj.(*apps.ReplicaSet)
		names = append(names, rs.Name)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(names, []string{"test-0", "test-1", "test-2", "test-3"}) {
		t.Errorf("unexpected items: %#v", list)
	}
}

func makeSecret(name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string][]byte{
			"key": []byte("value"),
		},
	}
}

func TestNameInFieldSelector(t *testing.T) {
	s, clientSet, closeFn := setup(t)
	defer closeFn()

	numNamespaces := 3
	for i := 0; i < 3; i++ {
		ns := framework.CreateTestingNamespace(fmt.Sprintf("ns%d", i), s, t)
		defer framework.DeleteTestingNamespace(ns, s, t)

		_, err := clientSet.CoreV1().Secrets(ns.Name).Create(context.TODO(), makeSecret("foo"), metav1.CreateOptions{})
		if err != nil {
			t.Errorf("Couldn't create secret: %v", err)
		}
		_, err = clientSet.CoreV1().Secrets(ns.Name).Create(context.TODO(), makeSecret("bar"), metav1.CreateOptions{})
		if err != nil {
			t.Errorf("Couldn't create secret: %v", err)
		}
	}

	testcases := []struct {
		namespace       string
		selector        string
		expectedSecrets int
	}{
		{
			namespace:       "",
			selector:        "metadata.name=foo",
			expectedSecrets: numNamespaces,
		},
		{
			namespace:       "",
			selector:        "metadata.name=foo,metadata.name=bar",
			expectedSecrets: 0,
		},
		{
			namespace:       "",
			selector:        "metadata.name=foo,metadata.namespace=ns1",
			expectedSecrets: 1,
		},
		{
			namespace:       "ns1",
			selector:        "metadata.name=foo,metadata.namespace=ns1",
			expectedSecrets: 1,
		},
		{
			namespace:       "ns1",
			selector:        "metadata.name=foo,metadata.namespace=ns2",
			expectedSecrets: 0,
		},
		{
			namespace:       "ns1",
			selector:        "metadata.name=foo,metadata.namespace=",
			expectedSecrets: 0,
		},
	}

	for _, tc := range testcases {
		opts := metav1.ListOptions{
			FieldSelector: tc.selector,
		}
		secrets, err := clientSet.CoreV1().Secrets(tc.namespace).List(context.TODO(), opts)
		if err != nil {
			t.Errorf("%s: Unexpected error: %v", tc.selector, err)
		}
		if len(secrets.Items) != tc.expectedSecrets {
			t.Errorf("%s: Unexpected number of secrets: %d, expected: %d", tc.selector, len(secrets.Items), tc.expectedSecrets)
		}
	}
}

type callWrapper struct {
	nested http.RoundTripper
	req    *http.Request
	resp   *http.Response
	err    error
}

func (w *callWrapper) RoundTrip(req *http.Request) (*http.Response, error) {
	w.req = req
	resp, err := w.nested.RoundTrip(req)
	w.resp = resp
	w.err = err
	return resp, err
}

func TestMetadataClient(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	s, clientset, closeFn := setup(t)
	defer closeFn()

	apiExtensionClient, err := apiextensionsclient.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	fooCRD := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos.cr.bar.com",
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "cr.bar.com",
			Version: "v1",
			Scope:   apiextensionsv1beta1.NamespaceScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural: "foos",
				Kind:   "Foo",
			},
			Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
				Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
			},
		},
	}
	fooCRD, err = fixtures.CreateNewCustomResourceDefinition(fooCRD, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}
	crdGVR := schema.GroupVersionResource{Group: fooCRD.Spec.Group, Version: fooCRD.Spec.Version, Resource: "foos"}

	testcases := []struct {
		name string
		want func(*testing.T)
	}{
		{
			name: "list, get, patch, and delete via metadata client",
			want: func(t *testing.T) {
				ns := "metadata-builtin"
				svc, err := clientset.CoreV1().Services(ns).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-1", Annotations: map[string]string{"foo": "bar"}}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create service: %v", err)
				}

				cfg := metadata.ConfigFor(&restclient.Config{Host: s.URL})
				wrapper := &callWrapper{}
				cfg.Wrap(func(rt http.RoundTripper) http.RoundTripper {
					wrapper.nested = rt
					return wrapper
				})

				client := metadata.NewForConfigOrDie(cfg).Resource(v1.SchemeGroupVersion.WithResource("services"))
				items, err := client.Namespace(ns).List(context.TODO(), metav1.ListOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if items.ResourceVersion == "" {
					t.Fatalf("unexpected items: %#v", items)
				}
				if len(items.Items) != 1 {
					t.Fatalf("unexpected list: %#v", items)
				}
				if item := items.Items[0]; item.Name != "test-1" || item.UID != svc.UID || item.Annotations["foo"] != "bar" {
					t.Fatalf("unexpected object: %#v", item)
				}

				if wrapper.resp == nil || wrapper.resp.Header.Get("Content-Type") != "application/vnd.kubernetes.protobuf" {
					t.Fatalf("unexpected response: %#v", wrapper.resp)
				}
				wrapper.resp = nil

				item, err := client.Namespace(ns).Get(context.TODO(), "test-1", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if item.ResourceVersion == "" || item.UID != svc.UID || item.Annotations["foo"] != "bar" {
					t.Fatalf("unexpected object: %#v", item)
				}
				if wrapper.resp == nil || wrapper.resp.Header.Get("Content-Type") != "application/vnd.kubernetes.protobuf" {
					t.Fatalf("unexpected response: %#v", wrapper.resp)
				}

				item, err = client.Namespace(ns).Patch(context.TODO(), "test-1", types.MergePatchType, []byte(`{"metadata":{"annotations":{"foo":"baz"}}}`), metav1.PatchOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if item.Annotations["foo"] != "baz" {
					t.Fatalf("unexpected object: %#v", item)
				}

				if err := client.Namespace(ns).Delete(context.TODO(), "test-1", metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &item.UID}}); err != nil {
					t.Fatal(err)
				}

				if _, err := client.Namespace(ns).Get(context.TODO(), "test-1", metav1.GetOptions{}); !apierrors.IsNotFound(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name: "list, get, patch, and delete via metadata client on a CRD",
			want: func(t *testing.T) {
				ns := "metadata-crd"
				crclient := dynamicClient.Resource(crdGVR).Namespace(ns)
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "cr.bar.com/v1",
						"kind":       "Foo",
						"spec":       map[string]interface{}{"field": 1},
						"metadata": map[string]interface{}{
							"name": "test-1",
							"annotations": map[string]interface{}{
								"foo": "bar",
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}

				cfg := metadata.ConfigFor(config)
				wrapper := &callWrapper{}
				cfg.Wrap(func(rt http.RoundTripper) http.RoundTripper {
					wrapper.nested = rt
					return wrapper
				})

				client := metadata.NewForConfigOrDie(cfg).Resource(crdGVR)
				items, err := client.Namespace(ns).List(context.TODO(), metav1.ListOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if items.ResourceVersion == "" {
					t.Fatalf("unexpected items: %#v", items)
				}
				if len(items.Items) != 1 {
					t.Fatalf("unexpected list: %#v", items)
				}
				if item := items.Items[0]; item.Name != "test-1" || item.UID != cr.GetUID() || item.Annotations["foo"] != "bar" {
					t.Fatalf("unexpected object: %#v", item)
				}

				if wrapper.resp == nil || wrapper.resp.Header.Get("Content-Type") != "application/vnd.kubernetes.protobuf" {
					t.Fatalf("unexpected response: %#v", wrapper.resp)
				}
				wrapper.resp = nil

				item, err := client.Namespace(ns).Get(context.TODO(), "test-1", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if item.ResourceVersion == "" || item.UID != cr.GetUID() || item.Annotations["foo"] != "bar" {
					t.Fatalf("unexpected object: %#v", item)
				}
				if wrapper.resp == nil || wrapper.resp.Header.Get("Content-Type") != "application/vnd.kubernetes.protobuf" {
					t.Fatalf("unexpected response: %#v", wrapper.resp)
				}

				item, err = client.Namespace(ns).Patch(context.TODO(), "test-1", types.MergePatchType, []byte(`{"metadata":{"annotations":{"foo":"baz"}}}`), metav1.PatchOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if item.Annotations["foo"] != "baz" {
					t.Fatalf("unexpected object: %#v", item)
				}

				if err := client.Namespace(ns).Delete(context.TODO(), "test-1", metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &item.UID}}); err != nil {
					t.Fatal(err)
				}
				if _, err := client.Namespace(ns).Get(context.TODO(), "test-1", metav1.GetOptions{}); !apierrors.IsNotFound(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name: "watch via metadata client",
			want: func(t *testing.T) {
				ns := "metadata-watch"
				svc, err := clientset.CoreV1().Services(ns).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-2", Annotations: map[string]string{"foo": "bar"}}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create service: %v", err)
				}
				if _, err := clientset.CoreV1().Services(ns).Patch(context.TODO(), "test-2", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}

				cfg := metadata.ConfigFor(&restclient.Config{Host: s.URL})
				wrapper := &callWrapper{}
				cfg.Wrap(func(rt http.RoundTripper) http.RoundTripper {
					wrapper.nested = rt
					return wrapper
				})

				client := metadata.NewForConfigOrDie(cfg).Resource(v1.SchemeGroupVersion.WithResource("services"))
				w, err := client.Namespace(ns).Watch(context.TODO(), metav1.ListOptions{ResourceVersion: svc.ResourceVersion, Watch: true})
				if err != nil {
					t.Fatal(err)
				}
				defer w.Stop()
				var r watch.Event
				select {
				case evt, ok := <-w.ResultChan():
					if !ok {
						t.Fatal("watch closed")
					}
					r = evt
				case <-time.After(5 * time.Second):
					t.Fatal("no watch event in 5 seconds, bug")
				}
				if r.Type != watch.Modified {
					t.Fatalf("unexpected watch: %#v", r)
				}
				item, ok := r.Object.(*metav1.PartialObjectMetadata)
				if !ok {
					t.Fatalf("unexpected object: %T", item)
				}
				if item.ResourceVersion == "" || item.Name != "test-2" || item.UID != svc.UID || item.Annotations["test"] != "1" {
					t.Fatalf("unexpected object: %#v", item)
				}

				if wrapper.resp == nil || wrapper.resp.Header.Get("Content-Type") != "application/vnd.kubernetes.protobuf;stream=watch" {
					t.Fatalf("unexpected response: %#v", wrapper.resp)
				}
			},
		},

		{
			name: "watch via metadata client on a CRD",
			want: func(t *testing.T) {
				ns := "metadata-watch-crd"
				crclient := dynamicClient.Resource(crdGVR).Namespace(ns)
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "cr.bar.com/v1",
						"kind":       "Foo",
						"spec":       map[string]interface{}{"field": 1},
						"metadata": map[string]interface{}{
							"name": "test-2",
							"annotations": map[string]interface{}{
								"foo": "bar",
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}

				cfg := metadata.ConfigFor(config)
				client := metadata.NewForConfigOrDie(cfg).Resource(crdGVR)

				patched, err := client.Namespace(ns).Patch(context.TODO(), "test-2", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if patched.GetResourceVersion() == cr.GetResourceVersion() {
					t.Fatalf("Patch did not modify object: %#v", patched)
				}

				wrapper := &callWrapper{}
				cfg.Wrap(func(rt http.RoundTripper) http.RoundTripper {
					wrapper.nested = rt
					return wrapper
				})
				client = metadata.NewForConfigOrDie(cfg).Resource(crdGVR)

				w, err := client.Namespace(ns).Watch(context.TODO(), metav1.ListOptions{ResourceVersion: cr.GetResourceVersion(), Watch: true})
				if err != nil {
					t.Fatal(err)
				}
				defer w.Stop()
				var r watch.Event
				select {
				case evt, ok := <-w.ResultChan():
					if !ok {
						t.Fatal("watch closed")
					}
					r = evt
				case <-time.After(5 * time.Second):
					t.Fatal("no watch event in 5 seconds, bug")
				}
				if r.Type != watch.Modified {
					t.Fatalf("unexpected watch: %#v", r)
				}
				item, ok := r.Object.(*metav1.PartialObjectMetadata)
				if !ok {
					t.Fatalf("unexpected object: %T", item)
				}
				if item.ResourceVersion == "" || item.Name != "test-2" || item.UID != cr.GetUID() || item.Annotations["test"] != "1" {
					t.Fatalf("unexpected object: %#v", item)
				}

				if wrapper.resp == nil || wrapper.resp.Header.Get("Content-Type") != "application/vnd.kubernetes.protobuf;stream=watch" {
					t.Fatalf("unexpected response: %#v", wrapper.resp)
				}
			},
		},
	}

	for i := range testcases {
		tc := testcases[i]
		t.Run(tc.name, func(t *testing.T) {
			tc.want(t)
		})
	}
}

func TestAPICRDProtobuf(t *testing.T) {
	testNamespace := "test-api-crd-protobuf"
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	s, _, closeFn := setup(t)
	defer closeFn()

	apiExtensionClient, err := apiextensionsclient.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	fooCRD := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos.cr.bar.com",
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "cr.bar.com",
			Version: "v1",
			Scope:   apiextensionsv1beta1.NamespaceScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural: "foos",
				Kind:   "Foo",
			},
			Subresources: &apiextensionsv1beta1.CustomResourceSubresources{Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{}},
		},
	}
	fooCRD, err = fixtures.CreateNewCustomResourceDefinition(fooCRD, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}
	crdGVR := schema.GroupVersionResource{Group: fooCRD.Spec.Group, Version: fooCRD.Spec.Version, Resource: "foos"}
	crclient := dynamicClient.Resource(crdGVR).Namespace(testNamespace)

	testcases := []struct {
		name        string
		accept      string
		subresource string
		object      func(*testing.T) (metav1.Object, string, string)
		wantErr     func(*testing.T, error)
		wantBody    func(*testing.T, io.Reader)
	}{
		{
			name:   "server returns 406 when asking for protobuf for CRDs, which dynamic client does not support",
			accept: "application/vnd.kubernetes.protobuf",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-1"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-1", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
				status := err.(apierrors.APIStatus).Status()
				data, _ := json.MarshalIndent(status, "", "  ")
				// because the dynamic client only has a json serializer, the client processing of the error cannot
				// turn the response into something meaningful, so we verify that fallback handling works correctly
				if !apierrors.IsUnexpectedServerError(err) {
					t.Fatal(string(data))
				}
				if status.Message != "the server was unable to respond with a content type that the client supports (get foos.cr.bar.com test-1)" {
					t.Fatal(string(data))
				}
			},
		},
		{
			name:   "server returns JSON when asking for protobuf and json for CRDs",
			accept: "application/vnd.kubernetes.protobuf,application/json",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "spec": map[string]interface{}{"field": 1}, "metadata": map[string]interface{}{"name": "test-2"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-2", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				obj := &unstructured.Unstructured{}
				if err := json.NewDecoder(w).Decode(obj); err != nil {
					t.Fatal(err)
				}
				v, ok, err := unstructured.NestedInt64(obj.UnstructuredContent(), "spec", "field")
				if !ok || err != nil {
					data, _ := json.MarshalIndent(obj.UnstructuredContent(), "", "  ")
					t.Fatalf("err=%v ok=%t json=%s", err, ok, string(data))
				}
				if v != 1 {
					t.Fatalf("unexpected body: %#v", obj.UnstructuredContent())
				}
			},
		},
		{
			name:        "server returns 406 when asking for protobuf for CRDs status, which dynamic client does not support",
			accept:      "application/vnd.kubernetes.protobuf",
			subresource: "status",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-3"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-3", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"3"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
				status := err.(apierrors.APIStatus).Status()
				data, _ := json.MarshalIndent(status, "", "  ")
				// because the dynamic client only has a json serializer, the client processing of the error cannot
				// turn the response into something meaningful, so we verify that fallback handling works correctly
				if !apierrors.IsUnexpectedServerError(err) {
					t.Fatal(string(data))
				}
				if status.Message != "the server was unable to respond with a content type that the client supports (get foos.cr.bar.com test-3)" {
					t.Fatal(string(data))
				}
			},
		},
		{
			name:        "server returns JSON when asking for protobuf and json for CRDs status",
			accept:      "application/vnd.kubernetes.protobuf,application/json",
			subresource: "status",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "spec": map[string]interface{}{"field": 1}, "metadata": map[string]interface{}{"name": "test-4"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-4", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"4"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				obj := &unstructured.Unstructured{}
				if err := json.NewDecoder(w).Decode(obj); err != nil {
					t.Fatal(err)
				}
				v, ok, err := unstructured.NestedInt64(obj.UnstructuredContent(), "spec", "field")
				if !ok || err != nil {
					data, _ := json.MarshalIndent(obj.UnstructuredContent(), "", "  ")
					t.Fatalf("err=%v ok=%t json=%s", err, ok, string(data))
				}
				if v != 1 {
					t.Fatalf("unexpected body: %#v", obj.UnstructuredContent())
				}
			},
		},
	}

	for i := range testcases {
		tc := testcases[i]
		t.Run(tc.name, func(t *testing.T) {
			obj, group, resource := tc.object(t)

			cfg := dynamic.ConfigFor(config)
			if len(group) == 0 {
				cfg = dynamic.ConfigFor(&restclient.Config{Host: s.URL})
				cfg.APIPath = "/api"
			} else {
				cfg.APIPath = "/apis"
			}
			cfg.GroupVersion = &schema.GroupVersion{Group: group, Version: "v1"}
			client, err := restclient.RESTClientFor(cfg)
			if err != nil {
				t.Fatal(err)
			}

			rv, _ := strconv.Atoi(obj.GetResourceVersion())
			if rv < 1 {
				rv = 1
			}

			w, err := client.Get().
				Resource(resource).NamespaceIfScoped(obj.GetNamespace(), len(obj.GetNamespace()) > 0).Name(obj.GetName()).SubResource(tc.subresource).
				SetHeader("Accept", tc.accept).
				Stream(context.TODO())
			if (tc.wantErr != nil) != (err != nil) {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantErr != nil {
				tc.wantErr(t, err)
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()
			tc.wantBody(t, w)
		})
	}
}

func TestTransform(t *testing.T) {
	testNamespace := "test-transform"
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	s, clientset, closeFn := setup(t)
	defer closeFn()

	apiExtensionClient, err := apiextensionsclient.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	fooCRD := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos.cr.bar.com",
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "cr.bar.com",
			Version: "v1",
			Scope:   apiextensionsv1beta1.NamespaceScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural: "foos",
				Kind:   "Foo",
			},
		},
	}
	fooCRD, err = fixtures.CreateNewCustomResourceDefinition(fooCRD, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}
	crdGVR := schema.GroupVersionResource{Group: fooCRD.Spec.Group, Version: fooCRD.Spec.Version, Resource: "foos"}
	crclient := dynamicClient.Resource(crdGVR).Namespace(testNamespace)

	testcases := []struct {
		name          string
		accept        string
		includeObject metav1.IncludeObjectPolicy
		object        func(*testing.T) (metav1.Object, string, string)
		wantErr       func(*testing.T, error)
		wantBody      func(*testing.T, io.Reader)
	}{
		{
			name:   "v1beta1 verify columns on cluster scoped resources",
			accept: "application/json;as=Table;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "default", Namespace: ""}, "", "namespaces"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableWatchEvents(t, 1, 3, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:   "v1beta1 verify columns on CRDs in json",
			accept: "application/json;as=Table;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-1"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-1", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableWatchEvents(t, 2, 2, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:   "v1beta1 verify columns on CRDs in json;stream=watch",
			accept: "application/json;stream=watch;as=Table;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-2"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-2", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableWatchEvents(t, 2, 2, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:   "v1beta1 verify columns on CRDs in yaml",
			accept: "application/yaml;as=Table;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-3"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-3", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
				// TODO: this should be a more specific error
				if err.Error() != "only the following media types are accepted: application/json;stream=watch, application/vnd.kubernetes.protobuf;stream=watch" {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1beta1 verify columns on services",
			accept: "application/json;as=Table;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				svc, err := clientset.CoreV1().Services(testNamespace).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-1"}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create service: %v", err)
				}
				if _, err := clientset.CoreV1().Services(testNamespace).Patch(context.TODO(), svc.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update service: %v", err)
				}
				return svc, "", "services"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableWatchEvents(t, 2, 7, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:          "v1beta1 verify columns on services with no object",
			accept:        "application/json;as=Table;g=meta.k8s.io;v=v1beta1",
			includeObject: metav1.IncludeNone,
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().Services(testNamespace).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-2"}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().Services(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "services"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableWatchEvents(t, 2, 7, metav1.IncludeNone, json.NewDecoder(w))
			},
		},
		{
			name:          "v1beta1 verify columns on services with full object",
			accept:        "application/json;as=Table;g=meta.k8s.io;v=v1beta1",
			includeObject: metav1.IncludeObject,
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().Services(testNamespace).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-3"}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().Services(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "services"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				objects := expectTableWatchEvents(t, 2, 7, metav1.IncludeObject, json.NewDecoder(w))
				var svc v1.Service
				if err := json.Unmarshal(objects[1], &svc); err != nil {
					t.Fatal(err)
				}
				if svc.Annotations["test"] != "1" || svc.Spec.Ports[0].Port != 1000 {
					t.Fatalf("unexpected object: %#v", svc)
				}
			},
		},
		{
			name:   "v1beta1 verify partial metadata object on config maps",
			accept: "application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().ConfigMaps(testNamespace).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "test-1", Annotations: map[string]string{"test": "0"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().ConfigMaps(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "configmaps"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectPartialObjectMetaEvents(t, json.NewDecoder(w), "0", "1")
			},
		},
		{
			name:   "v1beta1 verify partial metadata object on config maps in protobuf",
			accept: "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().ConfigMaps(testNamespace).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "test-2", Annotations: map[string]string{"test": "0"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().ConfigMaps(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "configmaps"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectPartialObjectMetaEventsProtobuf(t, w, "0", "1")
			},
		},
		{
			name:   "v1beta1 verify partial metadata object on CRDs in protobuf",
			accept: "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-4", "annotations": map[string]string{"test": "0"}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-4", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectPartialObjectMetaEventsProtobuf(t, w, "0", "1")
			},
		},
		{
			name:   "v1beta1 verify error on unsupported mimetype protobuf for table conversion",
			accept: "application/vnd.kubernetes.protobuf;as=Table;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
				// TODO: this should be a more specific error
				if err.Error() != "only the following media types are accepted: application/json, application/yaml, application/vnd.kubernetes.protobuf" {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "verify error on invalid mimetype - bad version",
			accept: "application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1alpha1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1beta1 verify error on invalid mimetype - bad group",
			accept: "application/json;as=PartialObjectMetadata;g=k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1beta1 verify error on invalid mimetype - bad kind",
			accept: "application/json;as=PartialObject;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1beta1 verify error on invalid mimetype - missing kind",
			accept: "application/json;g=meta.k8s.io;v=v1beta1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:          "v1beta1 verify error on invalid transform parameter",
			accept:        "application/json;as=Table;g=meta.k8s.io;v=v1beta1",
			includeObject: metav1.IncludeObjectPolicy("unrecognized"),
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsBadRequest(err) || !strings.Contains(err.Error(), `Invalid value: "unrecognized": must be 'Metadata', 'Object', 'None', or empty`) {
					t.Fatal(err)
				}
			},
		},

		{
			name:   "v1 verify columns on cluster scoped resources",
			accept: "application/json;as=Table;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "default", Namespace: ""}, "", "namespaces"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableV1WatchEvents(t, 1, 3, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:   "v1 verify columns on CRDs in json",
			accept: "application/json;as=Table;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-5"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-5", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableV1WatchEvents(t, 2, 2, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:   "v1 verify columns on CRDs in json;stream=watch",
			accept: "application/json;stream=watch;as=Table;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-6"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-6", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableV1WatchEvents(t, 2, 2, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:   "v1 verify columns on CRDs in yaml",
			accept: "application/yaml;as=Table;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-7"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), "test-7", types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
				// TODO: this should be a more specific error
				if err.Error() != "only the following media types are accepted: application/json;stream=watch, application/vnd.kubernetes.protobuf;stream=watch" {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1 verify columns on services",
			accept: "application/json;as=Table;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				svc, err := clientset.CoreV1().Services(testNamespace).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-5"}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create service: %v", err)
				}
				if _, err := clientset.CoreV1().Services(testNamespace).Patch(context.TODO(), svc.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update service: %v", err)
				}
				return svc, "", "services"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableV1WatchEvents(t, 2, 7, metav1.IncludeMetadata, json.NewDecoder(w))
			},
		},
		{
			name:          "v1 verify columns on services with no object",
			accept:        "application/json;as=Table;g=meta.k8s.io;v=v1",
			includeObject: metav1.IncludeNone,
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().Services(testNamespace).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-6"}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().Services(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "services"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectTableV1WatchEvents(t, 2, 7, metav1.IncludeNone, json.NewDecoder(w))
			},
		},
		{
			name:          "v1 verify columns on services with full object",
			accept:        "application/json;as=Table;g=meta.k8s.io;v=v1",
			includeObject: metav1.IncludeObject,
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().Services(testNamespace).Create(context.TODO(), &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "test-7"}, Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1000}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().Services(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "services"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				objects := expectTableV1WatchEvents(t, 2, 7, metav1.IncludeObject, json.NewDecoder(w))
				var svc v1.Service
				if err := json.Unmarshal(objects[1], &svc); err != nil {
					t.Fatal(err)
				}
				if svc.Annotations["test"] != "1" || svc.Spec.Ports[0].Port != 1000 {
					t.Fatalf("unexpected object: %#v", svc)
				}
			},
		},
		{
			name:   "v1 verify partial metadata object on config maps",
			accept: "application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().ConfigMaps(testNamespace).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "test-3", Annotations: map[string]string{"test": "0"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().ConfigMaps(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "configmaps"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectPartialObjectMetaV1Events(t, json.NewDecoder(w), "0", "1")
			},
		},
		{
			name:   "v1 verify partial metadata object on config maps in protobuf",
			accept: "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				obj, err := clientset.CoreV1().ConfigMaps(testNamespace).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "test-4", Annotations: map[string]string{"test": "0"}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create object: %v", err)
				}
				if _, err := clientset.CoreV1().ConfigMaps(testNamespace).Patch(context.TODO(), obj.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to update object: %v", err)
				}
				return obj, "", "configmaps"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectPartialObjectMetaV1EventsProtobuf(t, w, "0", "1")
			},
		},
		{
			name:   "v1 verify partial metadata object on CRDs in protobuf",
			accept: "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				cr, err := crclient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "cr.bar.com/v1", "kind": "Foo", "metadata": map[string]interface{}{"name": "test-8", "annotations": map[string]string{"test": "0"}}}}, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unable to create cr: %v", err)
				}
				if _, err := crclient.Patch(context.TODO(), cr.GetName(), types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}); err != nil {
					t.Fatalf("unable to patch cr: %v", err)
				}
				return cr, crdGVR.Group, "foos"
			},
			wantBody: func(t *testing.T, w io.Reader) {
				expectPartialObjectMetaV1EventsProtobuf(t, w, "0", "1")
			},
		},
		{
			name:   "v1 verify error on unsupported mimetype protobuf for table conversion",
			accept: "application/vnd.kubernetes.protobuf;as=Table;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
				// TODO: this should be a more specific error
				if err.Error() != "only the following media types are accepted: application/json, application/yaml, application/vnd.kubernetes.protobuf" {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1 verify error on invalid mimetype - bad group",
			accept: "application/json;as=PartialObjectMetadata;g=k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1 verify error on invalid mimetype - bad kind",
			accept: "application/json;as=PartialObject;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1 verify error on invalid mimetype - only meta kinds accepted",
			accept: "application/json;as=Service;g=;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:   "v1 verify error on invalid mimetype - missing kind",
			accept: "application/json;g=meta.k8s.io;v=v1",
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsNotAcceptable(err) {
					t.Fatal(err)
				}
			},
		},
		{
			name:          "v1 verify error on invalid transform parameter",
			accept:        "application/json;as=Table;g=meta.k8s.io;v=v1",
			includeObject: metav1.IncludeObjectPolicy("unrecognized"),
			object: func(t *testing.T) (metav1.Object, string, string) {
				return &metav1.ObjectMeta{Name: "kubernetes", Namespace: "default"}, "", "services"
			},
			wantErr: func(t *testing.T, err error) {
				if !apierrors.IsBadRequest(err) || !strings.Contains(err.Error(), `Invalid value: "unrecognized": must be 'Metadata', 'Object', 'None', or empty`) {
					t.Fatal(err)
				}
			},
		},
	}

	for i := range testcases {
		tc := testcases[i]
		t.Run(tc.name, func(t *testing.T) {
			obj, group, resource := tc.object(t)

			cfg := dynamic.ConfigFor(config)
			if len(group) == 0 {
				cfg = dynamic.ConfigFor(&restclient.Config{Host: s.URL})
				cfg.APIPath = "/api"
			} else {
				cfg.APIPath = "/apis"
			}
			cfg.GroupVersion = &schema.GroupVersion{Group: group, Version: "v1"}

			client, err := restclient.RESTClientFor(cfg)
			if err != nil {
				t.Fatal(err)
			}

			rv, _ := strconv.Atoi(obj.GetResourceVersion())
			if rv < 1 {
				rv = 1
			}

			w, err := client.Get().
				Resource(resource).NamespaceIfScoped(obj.GetNamespace(), len(obj.GetNamespace()) > 0).
				SetHeader("Accept", tc.accept).
				VersionedParams(&metav1.ListOptions{
					ResourceVersion: strconv.Itoa(rv - 1),
					Watch:           true,
					FieldSelector:   fields.OneTermEqualSelector("metadata.name", obj.GetName()).String(),
				}, metav1.ParameterCodec).
				Param("includeObject", string(tc.includeObject)).
				Stream(context.TODO())
			if (tc.wantErr != nil) != (err != nil) {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantErr != nil {
				tc.wantErr(t, err)
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()
			tc.wantBody(t, w)
		})
	}
}

func expectTableWatchEvents(t *testing.T, count, columns int, policy metav1.IncludeObjectPolicy, d *json.Decoder) [][]byte {
	t.Helper()

	var objects [][]byte

	for i := 0; i < count; i++ {
		var evt metav1.WatchEvent
		if err := d.Decode(&evt); err != nil {
			t.Fatal(err)
		}
		var table metav1beta1.Table
		if err := json.Unmarshal(evt.Object.Raw, &table); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			if len(table.ColumnDefinitions) != columns {
				t.Fatalf("Got unexpected columns on first watch event: %d vs %#v", columns, table.ColumnDefinitions)
			}
		} else {
			if len(table.ColumnDefinitions) != 0 {
				t.Fatalf("Expected no columns on second watch event: %#v", table.ColumnDefinitions)
			}
		}
		if len(table.Rows) != 1 {
			t.Fatalf("Invalid rows: %#v", table.Rows)
		}
		row := table.Rows[0]
		if len(row.Cells) != columns {
			t.Fatalf("Invalid row width: %#v", row.Cells)
		}
		switch policy {
		case metav1.IncludeMetadata:
			var meta metav1beta1.PartialObjectMetadata
			if err := json.Unmarshal(row.Object.Raw, &meta); err != nil {
				t.Fatalf("expected partial object: %v", err)
			}
			partialObj := metav1.TypeMeta{Kind: "PartialObjectMetadata", APIVersion: "meta.k8s.io/v1beta1"}
			if meta.TypeMeta != partialObj {
				t.Fatalf("expected partial object: %#v", meta)
			}
		case metav1.IncludeNone:
			if len(row.Object.Raw) != 0 {
				t.Fatalf("Expected no object: %s", string(row.Object.Raw))
			}
		case metav1.IncludeObject:
			if len(row.Object.Raw) == 0 {
				t.Fatalf("Expected object: %s", string(row.Object.Raw))
			}
			objects = append(objects, row.Object.Raw)
		}
	}
	return objects
}

func expectPartialObjectMetaEvents(t *testing.T, d *json.Decoder, values ...string) {
	t.Helper()

	for i, value := range values {
		var evt metav1.WatchEvent
		if err := d.Decode(&evt); err != nil {
			t.Fatal(err)
		}
		var meta metav1beta1.PartialObjectMetadata
		if err := json.Unmarshal(evt.Object.Raw, &meta); err != nil {
			t.Fatal(err)
		}
		typeMeta := metav1.TypeMeta{Kind: "PartialObjectMetadata", APIVersion: "meta.k8s.io/v1beta1"}
		if meta.TypeMeta != typeMeta {
			t.Fatalf("expected partial object: %#v", meta)
		}
		if meta.Annotations["test"] != value {
			t.Fatalf("expected event %d to have value %q instead of %q", i+1, value, meta.Annotations["test"])
		}
	}
}

func expectPartialObjectMetaEventsProtobuf(t *testing.T, r io.Reader, values ...string) {
	scheme := runtime.NewScheme()
	metav1.AddToGroupVersion(scheme, schema.GroupVersion{Version: "v1"})
	rs := protobuf.NewRawSerializer(scheme, scheme)
	d := streaming.NewDecoder(
		protobuf.LengthDelimitedFramer.NewFrameReader(ioutil.NopCloser(r)),
		rs,
	)
	ds := metainternalversionscheme.Codecs.UniversalDeserializer()

	for i, value := range values {
		var evt metav1.WatchEvent
		if _, _, err := d.Decode(nil, &evt); err != nil {
			t.Fatal(err)
		}
		obj, gvk, err := ds.Decode(evt.Object.Raw, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		meta, ok := obj.(*metav1beta1.PartialObjectMetadata)
		if !ok {
			t.Fatalf("unexpected watch object %T", obj)
		}
		expected := &schema.GroupVersionKind{Kind: "PartialObjectMetadata", Version: "v1beta1", Group: "meta.k8s.io"}
		if !reflect.DeepEqual(expected, gvk) {
			t.Fatalf("expected partial object: %#v", meta)
		}
		if meta.Annotations["test"] != value {
			t.Fatalf("expected event %d to have value %q instead of %q", i+1, value, meta.Annotations["test"])
		}
	}
}

func expectTableV1WatchEvents(t *testing.T, count, columns int, policy metav1.IncludeObjectPolicy, d *json.Decoder) [][]byte {
	t.Helper()

	var objects [][]byte

	for i := 0; i < count; i++ {
		var evt metav1.WatchEvent
		if err := d.Decode(&evt); err != nil {
			t.Fatal(err)
		}
		var table metav1.Table
		if err := json.Unmarshal(evt.Object.Raw, &table); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			if len(table.ColumnDefinitions) != columns {
				t.Fatalf("Got unexpected columns on first watch event: %d vs %#v", columns, table.ColumnDefinitions)
			}
		} else {
			if len(table.ColumnDefinitions) != 0 {
				t.Fatalf("Expected no columns on second watch event: %#v", table.ColumnDefinitions)
			}
		}
		if len(table.Rows) != 1 {
			t.Fatalf("Invalid rows: %#v", table.Rows)
		}
		row := table.Rows[0]
		if len(row.Cells) != columns {
			t.Fatalf("Invalid row width: %#v", row.Cells)
		}
		switch policy {
		case metav1.IncludeMetadata:
			var meta metav1.PartialObjectMetadata
			if err := json.Unmarshal(row.Object.Raw, &meta); err != nil {
				t.Fatalf("expected partial object: %v", err)
			}
			partialObj := metav1.TypeMeta{Kind: "PartialObjectMetadata", APIVersion: "meta.k8s.io/v1"}
			if meta.TypeMeta != partialObj {
				t.Fatalf("expected partial object: %#v", meta)
			}
		case metav1.IncludeNone:
			if len(row.Object.Raw) != 0 {
				t.Fatalf("Expected no object: %s", string(row.Object.Raw))
			}
		case metav1.IncludeObject:
			if len(row.Object.Raw) == 0 {
				t.Fatalf("Expected object: %s", string(row.Object.Raw))
			}
			objects = append(objects, row.Object.Raw)
		}
	}
	return objects
}

func expectPartialObjectMetaV1Events(t *testing.T, d *json.Decoder, values ...string) {
	t.Helper()

	for i, value := range values {
		var evt metav1.WatchEvent
		if err := d.Decode(&evt); err != nil {
			t.Fatal(err)
		}
		var meta metav1.PartialObjectMetadata
		if err := json.Unmarshal(evt.Object.Raw, &meta); err != nil {
			t.Fatal(err)
		}
		typeMeta := metav1.TypeMeta{Kind: "PartialObjectMetadata", APIVersion: "meta.k8s.io/v1"}
		if meta.TypeMeta != typeMeta {
			t.Fatalf("expected partial object: %#v", meta)
		}
		if meta.Annotations["test"] != value {
			t.Fatalf("expected event %d to have value %q instead of %q", i+1, value, meta.Annotations["test"])
		}
	}
}

func expectPartialObjectMetaV1EventsProtobuf(t *testing.T, r io.Reader, values ...string) {
	scheme := runtime.NewScheme()
	metav1.AddToGroupVersion(scheme, schema.GroupVersion{Version: "v1"})
	rs := protobuf.NewRawSerializer(scheme, scheme)
	d := streaming.NewDecoder(
		protobuf.LengthDelimitedFramer.NewFrameReader(ioutil.NopCloser(r)),
		rs,
	)
	ds := metainternalversionscheme.Codecs.UniversalDeserializer()

	for i, value := range values {
		var evt metav1.WatchEvent
		if _, _, err := d.Decode(nil, &evt); err != nil {
			t.Fatal(err)
		}
		obj, gvk, err := ds.Decode(evt.Object.Raw, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		meta, ok := obj.(*metav1.PartialObjectMetadata)
		if !ok {
			t.Fatalf("unexpected watch object %T", obj)
		}
		expected := &schema.GroupVersionKind{Kind: "PartialObjectMetadata", Version: "v1", Group: "meta.k8s.io"}
		if !reflect.DeepEqual(expected, gvk) {
			t.Fatalf("expected partial object: %#v", meta)
		}
		if meta.Annotations["test"] != value {
			t.Fatalf("expected event %d to have value %q instead of %q", i+1, value, meta.Annotations["test"])
		}
	}
}
