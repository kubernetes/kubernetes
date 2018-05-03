/*
Copyright 2018 The Kubernetes Authors.

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
	"path"
	"reflect"
	"testing"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/dynamic"
	"k8s.io/utils/pointer"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

func TestPruningCreate(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourcePruning, true)()

	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	crd.Spec.Prune = pointer.BoolPtr(true)
	crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'unspecified' field to be pruned")
	ns := "not-the-default"
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, crd)
	foo := newNoxuValidationInstance(ns, "foo")
	unstructured.SetNestedField(foo.Object, int64(42), "unspecified")
	obj, err := instantiateCustomResource(t, foo, noxuResourceClient, crd)
	if err != nil {
		t.Fatalf("Unable to create noxu instance: %v", err)
	}
	t.Logf("CR created: %#v", obj.UnstructuredContent())

	if _, found, _ := unstructured.NestedFieldNoCopy(obj.Object, "alpha"); !found {
		t.Errorf("Expected specified 'alpha' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(obj.Object, "unspecified"); found {
		t.Errorf("Expected 'unspecified' field to be pruned, but it was not")
	}
}

func TestPruningFromStorage(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourcePruning, true)()

	tearDown, config, options, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	serverConfig, err := options.Config()
	if err != nil {
		t.Fatal(err)
	}

	crd := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	crd.Spec.Prune = pointer.BoolPtr(true)
	crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionsClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	restOptions, err := serverConfig.GenericConfig.RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Spec.Names.Plural})
	if err != nil {
		t.Fatal(err)
	}
	tlsInfo := transport.TLSInfo{
		CertFile: restOptions.StorageConfig.CertFile,
		KeyFile:  restOptions.StorageConfig.KeyFile,
		CAFile:   restOptions.StorageConfig.CAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		t.Fatal(err)
	}
	etcdConfig := clientv3.Config{
		Endpoints: restOptions.StorageConfig.ServerList,
		TLS:       tlsConfig,
	}
	etcdclient, err := clientv3.New(etcdConfig)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating object with unknown field manually in etcd")

	original := fixtures.NewNoxuInstance("default", "foo")
	unstructured.SetNestedField(original.UnstructuredContent(), "bar", "foo")
	unstructured.SetNestedField(original.UnstructuredContent(), "abc", "alpha")
	unstructured.SetNestedField(original.UnstructuredContent(), float64(42), "beta")

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := path.Join("/", restOptions.StorageConfig.Prefix, crd.Spec.Group, "noxus/default/foo")
	val, _ := json.Marshal(original.UnstructuredContent())
	if _, err := etcdclient.Put(ctx, key, string(val)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Checking that CustomResource is pruned from unknown fields")

	noxuResourceClient := newNamespacedCustomResourceClient("default", dynamicClient, crd)
	obj, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if foo, found, err := unstructured.NestedFieldNoCopy(obj.UnstructuredContent(), "metadata", "foo"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if found {
		t.Errorf("unexpected to find foo=%#v", foo)
	}

	t.Logf("Checking that CustomResource has the known fields")

	if alpha, found, err := unstructured.NestedString(obj.UnstructuredContent(), "alpha"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if found && !reflect.DeepEqual(alpha, "abc") {
		t.Errorf("unexpected to find alpha=%#v", alpha)
	}
}

func TestPruningPatch(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourcePruning, true)()

	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := newNoxuValidationCRD(apiextensionsv1beta1.ClusterScoped)
	crd.Spec.Prune = pointer.BoolPtr(true)
	if crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, dynamicClient); err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuNamespacedResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, crd)

	cr := newNoxuValidationInstance(ns, "foo")
	if cr, err = noxuNamespacedResourceClient.Create(cr, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	// a patch with a change
	patch := []byte(`{"alpha": "def", "foo": "bar"}`)
	if cr, err = noxuNamespacedResourceClient.Patch("foo", types.MergePatchType, patch, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if alpha, ok := cr.UnstructuredContent()["alpha"]; !ok {
		t.Error("expected alpha to be created")
	} else if alpha != "def" {
		t.Errorf("unexpected value for alpha %q, expected %q", alpha, "def")
	}
	if foo, ok := cr.UnstructuredContent()["foo"]; ok {
		t.Errorf("expected foo to be pruned, got: %q", foo)
	}
}
