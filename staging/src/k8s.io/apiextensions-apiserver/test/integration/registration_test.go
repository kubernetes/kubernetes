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

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
)

func instantiateCustomResource(t *testing.T, instanceToCreate *unstructured.Unstructured, client dynamic.ResourceInterface, definition *apiextensionsv1beta1.CustomResourceDefinition) (*unstructured.Unstructured, error) {
	createdInstance, err := client.Create(instanceToCreate)
	if err != nil {
		t.Logf("%#v", createdInstance)
		return nil, err
	}
	createdObjectMeta, err := meta.Accessor(createdInstance)
	if err != nil {
		t.Fatal(err)
	}
	// it should have a UUID
	if len(createdObjectMeta.GetUID()) == 0 {
		t.Errorf("missing uuid: %#v", createdInstance)
	}
	createdTypeMeta, err := meta.TypeAccessor(createdInstance)
	if err != nil {
		t.Fatal(err)
	}
	if e, a := definition.Spec.Group+"/"+definition.Spec.Version, createdTypeMeta.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := definition.Spec.Names.Kind, createdTypeMeta.GetKind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	return createdInstance, nil
}

func NewNamespacedCustomResourceClient(ns string, client dynamic.Interface, definition *apiextensionsv1beta1.CustomResourceDefinition) dynamic.ResourceInterface {
	return client.Resource(&metav1.APIResource{
		Name:       definition.Spec.Names.Plural,
		Namespaced: definition.Spec.Scope == apiextensionsv1beta1.NamespaceScoped,
	}, ns)
}

func TestMultipleResourceInstances(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	ns := "not-the-default"
	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}
	noxuNamespacedResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	noxuList, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}

	noxuListListMeta, err := meta.ListAccessor(noxuList)
	if err != nil {
		t.Fatal(err)
	}

	noxuNamespacedWatch, err := noxuNamespacedResourceClient.Watch(metav1.ListOptions{ResourceVersion: noxuListListMeta.GetResourceVersion()})
	if err != nil {
		t.Fatal(err)
	}
	defer noxuNamespacedWatch.Stop()

	instances := map[string]*struct {
		Added    bool
		Deleted  bool
		Instance *unstructured.Unstructured
	}{
		"foo": {},
		"bar": {},
	}

	for key, val := range instances {
		val.Instance, err = instantiateCustomResource(t, testserver.NewNoxuInstance(ns, key), noxuNamespacedResourceClient, noxuDefinition)
		if err != nil {
			t.Fatalf("unable to create Noxu Instance %q:%v", key, err)
		}
	}

	addEvents := 0
	for addEvents < len(instances) {
		select {
		case watchEvent := <-noxuNamespacedWatch.ResultChan():
			if e, a := watch.Added, watchEvent.Type; e != a {
				t.Fatalf("expected %v, got %v", e, a)
			}
			name, err := meta.NewAccessor().Name(watchEvent.Object)
			if err != nil {
				t.Fatalf("unable to retrieve object name:%v", err)
			}
			if instances[name].Added {
				t.Fatalf("Add event already registered for %q", name)
			}
			instances[name].Added = true
			addEvents++
		case <-time.After(5 * time.Second):
			t.Fatalf("missing watch event")
		}
	}

	for key, val := range instances {
		gottenNoxuInstace, err := noxuNamespacedResourceClient.Get(key, metav1.GetOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if e, a := val.Instance, gottenNoxuInstace; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
	listWithItem, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := len(instances), len(listWithItem.(*unstructured.UnstructuredList).Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	for _, a := range listWithItem.(*unstructured.UnstructuredList).Items {
		if e := instances[a.GetName()].Instance; !reflect.DeepEqual(e, &a) {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
	for key := range instances {
		if err := noxuNamespacedResourceClient.Delete(key, nil); err != nil {
			t.Fatalf("unable to delete %s:%v", key, err)
		}
	}
	listWithoutItem, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(listWithoutItem.(*unstructured.UnstructuredList).Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	deleteEvents := 0
	for deleteEvents < len(instances) {
		select {
		case watchEvent := <-noxuNamespacedWatch.ResultChan():
			if e, a := watch.Deleted, watchEvent.Type; e != a {
				t.Errorf("expected %v, got %v", e, a)
				break
			}
			name, err := meta.NewAccessor().Name(watchEvent.Object)
			if err != nil {
				t.Errorf("unable to retrieve object name:%v", err)
			}
			if instances[name].Deleted {
				t.Errorf("Delete event already registered for %q", name)
			}
			instances[name].Deleted = true
			deleteEvents++
		case <-time.After(5 * time.Second):
			t.Errorf("missing watch event")
		}
	}
}

func TestMultipleRegistration(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	ns := "not-the-default"
	sameInstanceName := "foo"
	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}
	noxuNamespacedResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	createdNoxuInstance, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns, sameInstanceName), noxuNamespacedResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}

	gottenNoxuInstance, err := noxuNamespacedResourceClient.Get(sameInstanceName, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := createdNoxuInstance, gottenNoxuInstance; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	curletDefinition := testserver.NewCurletCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	curletVersionClient, err := testserver.CreateNewCustomResourceDefinition(curletDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}
	curletNamespacedResourceClient := NewNamespacedCustomResourceClient(ns, curletVersionClient, curletDefinition)
	createdCurletInstance, err := instantiateCustomResource(t, testserver.NewCurletInstance(ns, sameInstanceName), curletNamespacedResourceClient, curletDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}
	gottenCurletInstance, err := curletNamespacedResourceClient.Get(sameInstanceName, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := createdCurletInstance, gottenCurletInstance; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	// now re-GET noxu
	gottenNoxuInstance2, err := noxuNamespacedResourceClient.Get(sameInstanceName, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := createdNoxuInstance, gottenNoxuInstance2; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestDeRegistrationAndReRegistration(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)
	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	ns := "not-the-default"
	sameInstanceName := "foo"
	func() {
		noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
		if err != nil {
			t.Fatal(err)
		}
		noxuNamespacedResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
		if _, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns, sameInstanceName), noxuNamespacedResourceClient, noxuDefinition); err != nil {
			t.Fatal(err)
		}
		if err := testserver.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
		if _, err := testserver.GetCustomResourceDefinition(noxuDefinition, apiExtensionClient); err == nil || !errors.IsNotFound(err) {
			t.Fatalf("expected a NotFound error, got:%v", err)
		}
		if _, err = noxuNamespacedResourceClient.List(metav1.ListOptions{}); err == nil || !errors.IsNotFound(err) {
			t.Fatalf("expected a NotFound error, got:%v", err)
		}
		if _, err = noxuNamespacedResourceClient.Get("foo", metav1.GetOptions{}); err == nil || !errors.IsNotFound(err) {
			t.Fatalf("expected a NotFound error, got:%v", err)
		}
	}()

	func() {
		if _, err := testserver.GetCustomResourceDefinition(noxuDefinition, apiExtensionClient); err == nil || !errors.IsNotFound(err) {
			t.Fatalf("expected a NotFound error, got:%v", err)
		}
		noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
		if err != nil {
			t.Fatal(err)
		}
		noxuNamespacedResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
		initialList, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if _, err = noxuNamespacedResourceClient.Get(sameInstanceName, metav1.GetOptions{}); err == nil || !errors.IsNotFound(err) {
			t.Fatalf("expected a NotFound error, got:%v", err)
		}
		if e, a := 0, len(initialList.(*unstructured.UnstructuredList).Items); e != a {
			t.Fatalf("expected %v, got %v", e, a)
		}
		createdNoxuInstance, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns, sameInstanceName), noxuNamespacedResourceClient, noxuDefinition)
		if err != nil {
			t.Fatal(err)
		}
		gottenNoxuInstance, err := noxuNamespacedResourceClient.Get(sameInstanceName, metav1.GetOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if e, a := createdNoxuInstance, gottenNoxuInstance; !reflect.DeepEqual(e, a) {
			t.Fatalf("expected %v, got %v", e, a)
		}
		listWithItem, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if e, a := 1, len(listWithItem.(*unstructured.UnstructuredList).Items); e != a {
			t.Fatalf("expected %v, got %v", e, a)
		}
		if e, a := *createdNoxuInstance, listWithItem.(*unstructured.UnstructuredList).Items[0]; !reflect.DeepEqual(e, a) {
			t.Fatalf("expected %v, got %v", e, a)
		}

		if err := noxuNamespacedResourceClient.Delete(sameInstanceName, nil); err != nil {
			t.Fatal(err)
		}
		if _, err = noxuNamespacedResourceClient.Get(sameInstanceName, metav1.GetOptions{}); err == nil || !errors.IsNotFound(err) {
			t.Fatalf("expected a NotFound error, got:%v", err)
		}
		listWithoutItem, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if e, a := 0, len(listWithoutItem.(*unstructured.UnstructuredList).Items); e != a {
			t.Fatalf("expected %v, got %v", e, a)
		}
	}()
}

func TestEtcdStorage(t *testing.T) {
	config, err := testserver.DefaultServerConfig()
	if err != nil {
		t.Fatal(err)
	}
	stopCh, apiExtensionClient, clientPool, err := testserver.StartServer(config)
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	etcdPrefix := getPrefixFromConfig(t, config)

	ns1 := "another-default-is-possible"
	curletDefinition := testserver.NewCurletCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	curletVersionClient, err := testserver.CreateNewCustomResourceDefinition(curletDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}
	curletNamespacedResourceClient := NewNamespacedCustomResourceClient(ns1, curletVersionClient, curletDefinition)
	if _, err := instantiateCustomResource(t, testserver.NewCurletInstance(ns1, "bar"), curletNamespacedResourceClient, curletDefinition); err != nil {
		t.Fatalf("unable to create curlet cluster scoped Instance:%v", err)
	}

	ns2 := "the-cruel-default"
	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}
	noxuNamespacedResourceClient := NewNamespacedCustomResourceClient(ns2, noxuVersionClient, noxuDefinition)
	if _, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns2, "foo"), noxuNamespacedResourceClient, noxuDefinition); err != nil {
		t.Fatalf("unable to create noxu namespace scoped Instance:%v", err)
	}

	testcases := map[string]struct {
		etcdPath       string
		expectedObject *metaObject
	}{
		"namespacedNoxuDefinition": {
			etcdPath: "apiextensions.k8s.io/customresourcedefinitions/noxus.mygroup.example.com",
			expectedObject: &metaObject{
				Kind:       "CustomResourceDefinition",
				APIVersion: "apiextensions.k8s.io/v1beta1",
				Metadata: Metadata{
					Name:      "noxus.mygroup.example.com",
					Namespace: "",
					SelfLink:  "",
				},
			},
		},
		"namespacedNoxuInstance": {
			etcdPath: "mygroup.example.com/noxus/the-cruel-default/foo",
			expectedObject: &metaObject{
				Kind:       "WishIHadChosenNoxu",
				APIVersion: "mygroup.example.com/v1beta1",
				Metadata: Metadata{
					Name:      "foo",
					Namespace: "the-cruel-default",
					SelfLink:  "", // TODO double check: empty?
				},
			},
		},

		"clusteredCurletDefinition": {
			etcdPath: "apiextensions.k8s.io/customresourcedefinitions/curlets.mygroup.example.com",
			expectedObject: &metaObject{
				Kind:       "CustomResourceDefinition",
				APIVersion: "apiextensions.k8s.io/v1beta1",
				Metadata: Metadata{
					Name:      "curlets.mygroup.example.com",
					Namespace: "",
					SelfLink:  "",
				},
			},
		},

		"clusteredCurletInstance": {
			etcdPath: "mygroup.example.com/curlets/bar",
			expectedObject: &metaObject{
				Kind:       "Curlet",
				APIVersion: "mygroup.example.com/v1beta1",
				Metadata: Metadata{
					Name:      "bar",
					Namespace: "",
					SelfLink:  "", // TODO double check: empty?
				},
			},
		},
	}

	etcdURL, ok := os.LookupEnv("KUBE_INTEGRATION_ETCD_URL")
	if !ok {
		etcdURL = "http://127.0.0.1:2379"
	}
	cfg := clientv3.Config{
		Endpoints: []string{etcdURL},
	}
	c, err := clientv3.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	kv := clientv3.NewKV(c)
	for testName, tc := range testcases {
		output, err := getFromEtcd(kv, etcdPrefix, tc.etcdPath)
		if err != nil {
			t.Fatalf("%s - no path gotten from etcd:%v", testName, err)
		}
		if e, a := tc.expectedObject, output; !reflect.DeepEqual(e, a) {
			t.Errorf("%s - expected %#v\n got %#v\n", testName, e, a)
		}
	}
}

func getPrefixFromConfig(t *testing.T, config *extensionsapiserver.Config) string {
	extensionsOptionsGetter, ok := config.CRDRESTOptionsGetter.(extensionsapiserver.CRDRESTOptionsGetter)
	if !ok {
		t.Fatal("can't obtain etcd prefix: unable to cast config.CRDRESTOptionsGetter to extensionsapiserver.CRDRESTOptionsGetter")
	}
	return extensionsOptionsGetter.StoragePrefix
}

func getFromEtcd(keys clientv3.KV, prefix, localPath string) (*metaObject, error) {
	internalPath := path.Join("/", prefix, localPath) // TODO: Double check, should we concatenate two prefixes?
	response, err := keys.Get(context.Background(), internalPath)
	if err != nil {
		return nil, err
	}
	if response.More || response.Count != 1 || len(response.Kvs) != 1 {
		return nil, fmt.Errorf("Invalid etcd response (not found == %v): %#v", response.Count == 0, response)
	}
	obj := &metaObject{}
	if err := json.Unmarshal(response.Kvs[0].Value, obj); err != nil {
		return nil, err
	}
	return obj, nil
}

type metaObject struct {
	Kind       string `json:"kind,omitempty" protobuf:"bytes,1,opt,name=kind"`
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,2,opt,name=apiVersion"`
	Metadata   `json:"metadata,omitempty" protobuf:"bytes,3,opt,name=metadata"`
}

type Metadata struct {
	Name      string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,2,opt,name=namespace"`
	SelfLink  string `json:"selfLink,omitempty" protobuf:"bytes,3,opt,name=selfLink"`
}
