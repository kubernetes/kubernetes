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
	"context"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"testing"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/conversion"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/fake"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion"
	"k8s.io/apiextensions-apiserver/pkg/controller/establish"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/server/options"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/apiserver/pkg/util/webhook"
	utilpointer "k8s.io/utils/pointer"
)

func TestConvertFieldLabel(t *testing.T) {
	tests := []struct {
		name          string
		clusterScoped bool
		label         string
		expectError   bool
	}{
		{
			name:          "cluster scoped - name is ok",
			clusterScoped: true,
			label:         "metadata.name",
		},
		{
			name:          "cluster scoped - namespace is not ok",
			clusterScoped: true,
			label:         "metadata.namespace",
			expectError:   true,
		},
		{
			name:          "cluster scoped - other field is not ok",
			clusterScoped: true,
			label:         "some.other.field",
			expectError:   true,
		},
		{
			name:  "namespace scoped - name is ok",
			label: "metadata.name",
		},
		{
			name:  "namespace scoped - namespace is ok",
			label: "metadata.namespace",
		},
		{
			name:        "namespace scoped - other field is not ok",
			label:       "some.other.field",
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			crd := apiextensions.CustomResourceDefinition{
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: "None",
					},
				},
			}

			if test.clusterScoped {
				crd.Spec.Scope = apiextensions.ClusterScoped
			} else {
				crd.Spec.Scope = apiextensions.NamespaceScoped
			}
			f, err := conversion.NewCRConverterFactory(nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			_, c, err := f.NewConverter(&crd)

			label, value, err := c.ConvertFieldLabel(schema.GroupVersionKind{}, test.label, "value")
			if e, a := test.expectError, err != nil; e != a {
				t.Fatalf("err: expected %t, got %t", e, a)
			}
			if test.expectError {
				if e, a := "field label not supported: "+test.label, err.Error(); e != a {
					t.Errorf("err: expected %s, got %s", e, a)
				}
				return
			}

			if e, a := test.label, label; e != a {
				t.Errorf("label: expected %s, got %s", e, a)
			}
			if e, a := "value", value; e != a {
				t.Errorf("value: expected %s, got %s", e, a)
			}
		})
	}
}

func TestHandlerConversionWithWatchCache(t *testing.T) {
	testHandlerConversion(t, true)
}

func TestHandlerConversionWithoutWatchCache(t *testing.T) {
	testHandlerConversion(t, false)
}

func testHandlerConversion(t *testing.T, enableWatchCache bool) {
	cl := fake.NewSimpleClientset()
	informers := informers.NewSharedInformerFactory(fake.NewSimpleClientset(), 0)
	crdInformer := informers.Apiextensions().InternalVersion().CustomResourceDefinitions()

	server, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)
	defer server.Terminate(t)

	crd := multiVersionFixture.DeepCopy()
	if _, err := cl.Apiextensions().CustomResourceDefinitions().Create(crd); err != nil {
		t.Fatal(err)
	}
	if err := crdInformer.Informer().GetStore().Add(crd); err != nil {
		t.Fatal(err)
	}

	etcdOptions := options.NewEtcdOptions(storageConfig)
	etcdOptions.StorageConfig.Codec = unstructured.UnstructuredJSONScheme
	restOptionsGetter := generic.RESTOptions{
		StorageConfig:           &etcdOptions.StorageConfig,
		Decorator:               generic.UndecoratedStorage,
		EnableGarbageCollection: true,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          crd.Spec.Group + "/" + crd.Spec.Names.Plural,
		CountMetricPollPeriod:   time.Minute,
	}
	if enableWatchCache {
		restOptionsGetter.Decorator = genericregistry.StorageWithCacher(100)
	}

	handler, err := NewCustomResourceDefinitionHandler(
		&versionDiscoveryHandler{}, &groupDiscoveryHandler{},
		crdInformer,
		http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
		restOptionsGetter,
		dummyAdmissionImpl{},
		&establish.EstablishingController{},
		dummyServiceResolverImpl{},
		func(r webhook.AuthenticationInfoResolver) webhook.AuthenticationInfoResolver { return r },
		1,
		dummyAuthorizerImpl{},
		time.Minute, 3*1024*1024)
	if err != nil {
		t.Fatal(err)
	}

	crdInfo, err := handler.getOrCreateServingInfoFor(crd.UID, crd.Name)
	if err != nil {
		t.Fatal(err)
	}

	updateValidateFunc := func(obj, old runtime.Object) error { return nil }
	validateFunc := func(obj runtime.Object) error { return nil }
	startResourceVersion := ""

	if enableWatchCache {
		// Let watch cache establish initial list
		time.Sleep(time.Second)
	}

	// Create and delete a marker object to get a starting resource version
	{
		u := &unstructured.Unstructured{Object: map[string]interface{}{}}
		u.SetGroupVersionKind(schema.GroupVersionKind{Group: "stable.example.com", Version: "v1beta1", Kind: "MultiVersion"})
		u.SetName("marker")
		if item, err := crdInfo.storages["v1beta1"].CustomResource.Create(context.TODO(), u, validateFunc, &metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		} else {
			startResourceVersion = item.(*unstructured.Unstructured).GetResourceVersion()
		}
		if _, _, err := crdInfo.storages["v1beta1"].CustomResource.Delete(context.TODO(), u.GetName(), validateFunc, &metav1.DeleteOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	// Create and get every version, expect returned result to match creation GVK
	for _, version := range crd.Spec.Versions {
		expectGVK := schema.GroupVersionKind{Group: "stable.example.com", Version: version.Name, Kind: "MultiVersion"}
		u := &unstructured.Unstructured{Object: map[string]interface{}{}}
		u.SetGroupVersionKind(expectGVK)
		u.SetName("my-" + version.Name)
		unstructured.SetNestedField(u.Object, int64(1), "spec", "num")

		// Create
		if item, err := crdInfo.storages[version.Name].CustomResource.Create(context.TODO(), u, validateFunc, &metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		} else if item.GetObjectKind().GroupVersionKind() != expectGVK {
			t.Errorf("expected create result to be %#v, got %#v", expectGVK, item.GetObjectKind().GroupVersionKind())
		} else {
			u = item.(*unstructured.Unstructured)
		}

		// Update
		u.SetAnnotations(map[string]string{"updated": "true"})
		if item, _, err := crdInfo.storages[version.Name].CustomResource.Update(context.TODO(), u.GetName(), rest.DefaultUpdatedObjectInfo(u), validateFunc, updateValidateFunc, false, &metav1.UpdateOptions{}); err != nil {
			t.Fatal(err)
		} else if item.GetObjectKind().GroupVersionKind() != expectGVK {
			t.Errorf("expected update result to be %#v, got %#v", expectGVK, item.GetObjectKind().GroupVersionKind())
		}

		// Get
		if item, err := crdInfo.storages[version.Name].CustomResource.Get(context.TODO(), u.GetName(), &metav1.GetOptions{}); err != nil {
			t.Fatal(err)
		} else if item.GetObjectKind().GroupVersionKind() != expectGVK {
			t.Errorf("expected get result to be %#v, got %#v", expectGVK, item.GetObjectKind().GroupVersionKind())
		}

		if enableWatchCache {
			// Allow time to propagate the create into the cache
			time.Sleep(time.Second)
			// Get cached
			if item, err := crdInfo.storages[version.Name].CustomResource.Get(context.TODO(), u.GetName(), &metav1.GetOptions{ResourceVersion: "0"}); err != nil {
				t.Fatal(err)
			} else if item.GetObjectKind().GroupVersionKind() != expectGVK {
				t.Errorf("expected cached get result to be %#v, got %#v", expectGVK, item.GetObjectKind().GroupVersionKind())
			}
		}
	}

	// List every version, expect all returned items to match request GVK
	for _, version := range crd.Spec.Versions {
		expectGVK := schema.GroupVersionKind{Group: "stable.example.com", Version: version.Name, Kind: "MultiVersion"}

		if list, err := crdInfo.storages[version.Name].CustomResource.List(context.TODO(), &metainternalversion.ListOptions{}); err != nil {
			t.Fatal(err)
		} else {
			for _, item := range list.(*unstructured.UnstructuredList).Items {
				if item.GroupVersionKind() != expectGVK {
					t.Errorf("expected list item to be %#v, got %#v", expectGVK, item.GroupVersionKind())
				}
			}
		}

		if enableWatchCache {
			// List from watch cache
			if list, err := crdInfo.storages[version.Name].CustomResource.List(context.TODO(), &metainternalversion.ListOptions{ResourceVersion: "0"}); err != nil {
				t.Fatal(err)
			} else {
				for _, item := range list.(*unstructured.UnstructuredList).Items {
					if item.GroupVersionKind() != expectGVK {
						t.Errorf("expected cached list item to be %#v, got %#v", expectGVK, item.GroupVersionKind())
					}
				}
			}
		}

		watch, err := crdInfo.storages[version.Name].CustomResource.Watch(context.TODO(), &metainternalversion.ListOptions{ResourceVersion: startResourceVersion})
		if err != nil {
			t.Fatal(err)
		}
		// 5 events: delete marker, create v1alpha1, create v1beta1, update v1alpha1, update v1beta1
		for i := 0; i < 5; i++ {
			select {
			case event, ok := <-watch.ResultChan():
				if !ok {
					t.Fatalf("watch closed")
				}
				item, isUnstructured := event.Object.(*unstructured.Unstructured)
				if !isUnstructured {
					t.Fatalf("unexpected object type %T: %#v", item, event)
				}
				if item.GroupVersionKind() != expectGVK {
					t.Errorf("expected watch object to be %#v, got %#v", expectGVK, item.GroupVersionKind())
				}
			case <-time.After(time.Second):
				t.Errorf("timed out waiting for watch event")
			}
		}
		// Expect no more watch events
		select {
		case event := <-watch.ResultChan():
			t.Errorf("unexpected event: %#v", event)
		case <-time.After(time.Second):
		}
	}
}

type dummyAdmissionImpl struct{}

func (dummyAdmissionImpl) Handles(operation admission.Operation) bool { return false }

type dummyAuthorizerImpl struct{}

func (dummyAuthorizerImpl) Authorize(a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return authorizer.DecisionAllow, "", nil
}

type dummyServiceResolverImpl struct{}

func (dummyServiceResolverImpl) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return &url.URL{Scheme: "https", Host: net.JoinHostPort(name+"."+namespace+".svc", strconv.Itoa(int(port)))}, nil
}

var multiVersionFixture = &apiextensions.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "multiversion.stable.example.com", UID: types.UID("12345")},
	Spec: apiextensions.CustomResourceDefinitionSpec{
		Group: "stable.example.com",
		Names: apiextensions.CustomResourceDefinitionNames{
			Plural: "multiversion", Singular: "multiversion", Kind: "MultiVersion", ShortNames: []string{"mv"}, ListKind: "MultiVersionList", Categories: []string{"all"},
		},
		Conversion:            &apiextensions.CustomResourceConversion{Strategy: apiextensions.NoneConverter},
		Scope:                 apiextensions.ClusterScoped,
		PreserveUnknownFields: utilpointer.BoolPtr(false),
		Versions: []apiextensions.CustomResourceDefinitionVersion{
			{
				// storage version, same schema as v1alpha1
				Name: "v1beta1", Served: true, Storage: true,
				Subresources: &apiextensions.CustomResourceSubresources{Status: &apiextensions.CustomResourceSubresourceStatus{}},
				Schema: &apiextensions.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
						Type:       "object",
						Properties: map[string]apiextensions.JSONSchemaProps{"num": {Type: "integer", Description: "v1beta1 num field"}},
					},
				},
			},
			{
				// same schema as v1beta1
				Name: "v1alpha1", Served: true, Storage: false,
				Subresources: &apiextensions.CustomResourceSubresources{Status: &apiextensions.CustomResourceSubresourceStatus{}},
				Schema: &apiextensions.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
						Type:       "object",
						Properties: map[string]apiextensions.JSONSchemaProps{"num": {Type: "integer", Description: "v1alpha1 num field"}},
					},
				},
			},
		},
	},
	Status: apiextensions.CustomResourceDefinitionStatus{
		AcceptedNames: apiextensions.CustomResourceDefinitionNames{
			Plural: "multiversion", Singular: "multiversion", Kind: "MultiVersion", ShortNames: []string{"mv"}, ListKind: "MultiVersionList", Categories: []string{"all"},
		},
	},
}
