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
	"context"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"

	"sigs.k8s.io/yaml"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/dynamic"

	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

var pruningFixture = &apiextensionsv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "foos.tests.example.com"},
	Spec: apiextensionsv1.CustomResourceDefinitionSpec{
		Group: "tests.example.com",
		Scope: apiextensionsv1.ClusterScoped,
		Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1beta1",
				Served:  true,
				Storage: true,
				Subresources: &apiextensionsv1.CustomResourceSubresources{
					Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
				},
				Schema: &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						Type: "object",
					},
				},
			},
		},
		Names: apiextensionsv1.CustomResourceDefinitionNames{
			Plural:   "foos",
			Singular: "foo",
			Kind:     "Foo",
			ListKind: "FooList",
		},
	},
}

const (
	fooSchema = `
type: object
properties:
  alpha:
    type: string
  beta:
    type: number
`

	fooStatusSchema = `
type: object
properties:
  status:
    type: object
    properties:
      alpha:
        type: string
      beta:
        type: number
`

	fooSchemaPreservingUnknownFields = `
type: object
properties:
  alpha:
    type: string
  beta:
    type: number
  preserving:
    type: object
    x-kubernetes-preserve-unknown-fields: true
    properties:
      preserving:
        type: object
        x-kubernetes-preserve-unknown-fields: true
      pruning:
        type: object
  pruning:
    type: object
    properties:
      preserving:
        type: object
        x-kubernetes-preserve-unknown-fields: true
      pruning:
        type: object
x-kubernetes-preserve-unknown-fields: true
`

	fooSchemaEmbeddedResource = `
type: object
properties:
  embeddedPruning:
    type: object
    x-kubernetes-embedded-resource: true
    properties:
      specified:
        type: string
  embeddedPreserving:
    type: object
    x-kubernetes-embedded-resource: true
    x-kubernetes-preserve-unknown-fields: true
  embeddedNested:
    type: object
    x-kubernetes-embedded-resource: true
    x-kubernetes-preserve-unknown-fields: true
    properties:
      embeddedPruning:
        type: object
        x-kubernetes-embedded-resource: true
        properties:
          specified:
            type: string
`

	fooSchemaEmbeddedResourceInstance = pruningFooInstance + `
embeddedPruning:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
    unspecified: bar
  unspecified: bar
  specified: bar
embeddedPreserving:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
    unspecified: bar
  unspecified: bar
embeddedNested:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
    unspecified: bar
  unspecified: bar
  embeddedPruning:
    apiVersion: foo/v1
    kind: Foo
    metadata:
      name: foo
      unspecified: bar
    unspecified: bar
    specified: bar
`

	pruningFooInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
metadata:
  name: foo
`
)

func TestPruningCreate(t *testing.T) {
	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := pruningFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(fooSchema), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'unspecified' fields to be pruned")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural})
	foo := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(pruningFooInstance), &foo.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(foo.Object, "bar", "unspecified")
	unstructured.SetNestedField(foo.Object, "abc", "alpha")
	unstructured.SetNestedField(foo.Object, float64(42.0), "beta")
	unstructured.SetNestedField(foo.Object, "bar", "metadata", "unspecified")
	unstructured.SetNestedField(foo.Object, "bar", "metadata", "labels", "foo")
	foo, err = fooClient.Create(context.TODO(), foo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	t.Logf("CR created: %#v", foo.UnstructuredContent())

	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "unspecified"); found {
		t.Errorf("Expected 'unspecified' field to be pruned, but it was not")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "alpha"); !found {
		t.Errorf("Expected specified 'alpha' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "beta"); !found {
		t.Errorf("Expected specified 'beta' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "metadata", "unspecified"); found {
		t.Errorf("Expected 'metadata.unspecified' field to be pruned, but it was not")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "metadata", "labels", "foo"); !found {
		t.Errorf("Expected specified 'metadata.labels[foo]' field to stay, but it was pruned")
	}
}

func TestPruningStatus(t *testing.T) {
	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := pruningFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(fooStatusSchema), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'unspecified' fields to be pruned")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural})
	foo := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(pruningFooInstance), &foo.Object); err != nil {
		t.Fatal(err)
	}
	foo, err = fooClient.Create(context.TODO(), foo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	t.Logf("CR created: %#v", foo.UnstructuredContent())

	unstructured.SetNestedField(foo.Object, "bar", "status", "unspecified")
	unstructured.SetNestedField(foo.Object, "abc", "status", "alpha")
	unstructured.SetNestedField(foo.Object, float64(42.0), "status", "beta")
	unstructured.SetNestedField(foo.Object, "bar", "metadata", "unspecified")

	foo, err = fooClient.UpdateStatus(context.TODO(), foo, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unable to update status: %v", err)
	}

	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "unspecified"); found {
		t.Errorf("Expected 'status.unspecified' field to be pruned, but it was not")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "status", "alpha"); !found {
		t.Errorf("Expected specified 'status.alpha' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "status", "beta"); !found {
		t.Errorf("Expected specified 'status.beta' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "metadata", "unspecified"); found {
		t.Errorf("Expected 'metadata.unspecified' field to be pruned, but it was not")
	}
}

func TestPruningFromStorage(t *testing.T) {
	tearDown, config, completedConfig, err := fixtures.StartDefaultServerWithConfigAccess(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	crd := pruningFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(fooSchema), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	restOptions, err := completedConfig.GenericConfig.RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Spec.Names.Plural}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tlsInfo := transport.TLSInfo{
		CertFile:      restOptions.StorageConfig.Transport.CertFile,
		KeyFile:       restOptions.StorageConfig.Transport.KeyFile,
		TrustedCAFile: restOptions.StorageConfig.Transport.TrustedCAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		t.Fatal(err)
	}
	etcdConfig := clientv3.Config{
		Endpoints:   restOptions.StorageConfig.Transport.ServerList,
		DialTimeout: 20 * time.Second,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: tlsConfig,
	}
	etcdclient, err := clientv3.New(etcdConfig)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating object with unknown field manually in etcd")

	original := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(pruningFooInstance), &original.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(original.Object, "bar", "unspecified")
	unstructured.SetNestedField(original.Object, "abc", "alpha")
	unstructured.SetNestedField(original.Object, float64(42), "beta")
	unstructured.SetNestedField(original.Object, "bar", "metadata", "labels", "foo")

	// Note: we don't add metadata.unspecified as in the other tests. ObjectMeta pruning is independent of the generic pruning
	//       and we do not guarantee that we prune ObjectMeta on read from etcd.

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := path.Join("/", restOptions.StorageConfig.Prefix, crd.Spec.Group, "foos/foo")
	val, _ := json.Marshal(original.UnstructuredContent())
	if _, err := etcdclient.Put(ctx, key, string(val)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Checking that CustomResource is pruned from unknown fields")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural})
	foo, err := fooClient.Get(context.TODO(), "foo", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "unspecified"); found {
		t.Errorf("Expected 'unspecified' field to be pruned, but it was not")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "alpha"); !found {
		t.Errorf("Expected specified 'alpha' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "beta"); !found {
		t.Errorf("Expected specified 'beta' field to stay, but it was pruned")
	}

	// Note: we don't check metadata.foo as in the other tests. ObjectMeta pruning is independent of the generic pruning
	//       and we do not guarantee that we prune ObjectMeta on read from etcd.

	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "metadata", "labels", "foo"); !found {
		t.Errorf("Expected specified 'metadata.labels[foo]' field to stay, but it was pruned")
	}
}

func TestPruningPatch(t *testing.T) {
	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := pruningFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(fooSchema), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}
	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	fooClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural})
	foo := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(pruningFooInstance), &foo.Object); err != nil {
		t.Fatal(err)
	}
	foo, err = fooClient.Create(context.TODO(), foo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	t.Logf("CR created: %#v", foo.UnstructuredContent())

	// a patch with a change
	patch := []byte(`{"alpha": "abc", "beta": 42.0, "unspecified": "bar", "metadata": {"unspecified": "bar", "labels":{"foo":"bar"}}}`)
	if foo, err = fooClient.Patch(context.TODO(), "foo", types.MergePatchType, patch, metav1.PatchOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "unspecified"); found {
		t.Errorf("Expected 'unspecified' field to be pruned, but it was not")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "alpha"); !found {
		t.Errorf("Expected specified 'alpha' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "beta"); !found {
		t.Errorf("Expected specified 'beta' field to stay, but it was pruned")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "metadata", "unspecified"); found {
		t.Errorf("Expected 'metadata.unspecified' field to be pruned, but it was not")
	}
	if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, "metadata", "labels", "foo"); !found {
		t.Errorf("Expected specified 'metadata.labels[foo]' field to stay, but it was pruned")
	}
}

func TestPruningCreatePreservingUnknownFields(t *testing.T) {
	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := pruningFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(fooSchemaPreservingUnknownFields), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'unspecified' field to be pruned")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural})
	foo := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(pruningFooInstance), &foo.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(foo.Object, "bar", "unspecified")
	unstructured.SetNestedField(foo.Object, "abc", "alpha")
	unstructured.SetNestedField(foo.Object, float64(42.0), "beta")
	unstructured.SetNestedField(foo.Object, "bar", "metadata", "unspecified")
	unstructured.SetNestedField(foo.Object, "bar", "metadata", "labels", "foo")
	unstructured.SetNestedField(foo.Object, map[string]interface{}{
		"unspecified":       "bar",
		"unspecifiedObject": map[string]interface{}{"unspecified": "bar"},
		"pruning":           map[string]interface{}{"unspecified": "bar"},
		"preserving":        map[string]interface{}{"unspecified": "bar"},
	}, "pruning")
	unstructured.SetNestedField(foo.Object, map[string]interface{}{
		"unspecified":       "bar",
		"unspecifiedObject": map[string]interface{}{"unspecified": "bar"},
		"pruning":           map[string]interface{}{"unspecified": "bar"},
		"preserving":        map[string]interface{}{"unspecified": "bar"},
	}, "preserving")

	foo, err = fooClient.Create(context.TODO(), foo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	t.Logf("CR created: %#v", foo.UnstructuredContent())

	for _, pth := range [][]string{
		{"unspecified"},
		{"alpha"},
		{"beta"},
		{"metadata", "labels", "foo"},

		{"pruning", "pruning"},
		{"pruning", "preserving"},
		{"pruning", "preserving", "unspecified"},

		{"preserving", "unspecified"},
		{"preserving", "unspecifiedObject"},
		{"preserving", "unspecifiedObject", "unspecified"},
		{"preserving", "pruning"},
		{"preserving", "preserving"},
		{"preserving", "preserving", "unspecified"},
	} {
		if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, pth...); !found {
			t.Errorf("Expected '%s' field to stay, but it was pruned", strings.Join(pth, "."))
		}
	}
	for _, pth := range [][]string{
		{"metadata", "unspecified"},

		{"pruning", "unspecified"},
		{"pruning", "unspecifiedObject"},
		{"pruning", "unspecifiedObject", "unspecified"},
		{"pruning", "pruning", "unspecified"},

		{"preserving", "pruning", "unspecified"},
	} {
		if _, found, _ := unstructured.NestedFieldNoCopy(foo.Object, pth...); found {
			t.Errorf("Expected '%s' field to be pruned, but it was not", strings.Join(pth, "."))
		}
	}
}

func TestPruningEmbeddedResources(t *testing.T) {
	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := pruningFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(fooSchemaEmbeddedResource), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'unspecified' field to be pruned")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural})
	foo := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(fooSchemaEmbeddedResourceInstance), &foo.Object); err != nil {
		t.Fatal(err)
	}
	foo, err = fooClient.Create(context.TODO(), foo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	t.Logf("CR created: %#v", foo.UnstructuredContent())

	t.Logf("Comparing with expected, pruned value")
	x := runtime.DeepCopyJSON(foo.Object)
	delete(x, "apiVersion")
	delete(x, "kind")
	delete(x, "metadata")
	var expected map[string]interface{}
	if err := yaml.Unmarshal([]byte(`
embeddedPruning:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
  specified: bar
embeddedPreserving:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
  unspecified: bar
embeddedNested:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
  embeddedPruning:
    apiVersion: foo/v1
    kind: Foo
    metadata:
      name: foo
    specified: bar
  unspecified: bar
`), &expected); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, x) {
		t.Errorf("unexpected diff: %s", cmp.Diff(expected, x))
	}
}
