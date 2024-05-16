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
	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"sigs.k8s.io/yaml"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/dynamic"
	"k8s.io/utils/pointer"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	serveroptions "k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

func TestPostInvalidObjectMeta(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	noxuResourceClient := newNamespacedCustomResourceClient("default", dynamicClient, noxuDefinition)

	obj := fixtures.NewNoxuInstance("default", "foo")
	unstructured.SetNestedField(obj.UnstructuredContent(), int64(42), "metadata", "unknown")
	unstructured.SetNestedField(obj.UnstructuredContent(), nil, "metadata", "generation")
	unstructured.SetNestedField(obj.UnstructuredContent(), map[string]interface{}{"foo": int64(42), "bar": "abc"}, "metadata", "labels")
	_, err = instantiateCustomResource(t, obj, noxuResourceClient, noxuDefinition)
	if err == nil {
		t.Fatalf("unexpected non-error, expected invalid labels to be rejected: %v", err)
	}
	if status, ok := err.(errors.APIStatus); !ok {
		t.Fatalf("expected APIStatus error, but got: %#v", err)
	} else if !errors.IsBadRequest(err) {
		t.Fatalf("expected BadRequst error, but got: %v", errors.ReasonForError(err))
	} else if !strings.Contains(status.Status().Message, "cannot be handled") {
		t.Fatalf("expected 'cannot be handled' error message, got: %v", status.Status().Message)
	}

	unstructured.SetNestedField(obj.UnstructuredContent(), map[string]interface{}{"bar": "abc"}, "metadata", "labels")
	obj, err = instantiateCustomResource(t, obj, noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if unknown, found, err := unstructured.NestedInt64(obj.UnstructuredContent(), "metadata", "unknown"); err != nil {
		t.Errorf("unexpected error getting metadata.unknown: %v", err)
	} else if found {
		t.Errorf("unexpected metadata.unknown=%#v: expected this to be pruned", unknown)
	}

	if generation, found, err := unstructured.NestedInt64(obj.UnstructuredContent(), "metadata", "generation"); err != nil {
		t.Errorf("unexpected error getting metadata.generation: %v", err)
	} else if !found {
		t.Errorf("expected metadata.generation=1: got: %d", generation)
	} else if generation != 1 {
		t.Errorf("unexpected metadata.generation=%d: expected this to be set to 1", generation)
	}
}

func TestInvalidObjectMetaInStorage(t *testing.T) {
	tearDown, config, options, err := fixtures.StartDefaultServer(t)
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

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)
	noxuDefinition.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{
		OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
			Type: "object",
			Properties: map[string]apiextensionsv1.JSONSchemaProps{
				"embedded": {
					Type:                   "object",
					XEmbeddedResource:      true,
					XPreserveUnknownFields: pointer.BoolPtr(true),
				},
			},
		},
	}
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	RESTOptionsGetter := serveroptions.NewCRDRESTOptionsGetter(*options.RecommendedOptions.Etcd, nil, nil)
	restOptions, err := RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: noxuDefinition.Spec.Group, Resource: noxuDefinition.Spec.Names.Plural})
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

	t.Logf("Creating object with wrongly typed annotations and non-validating labels manually in etcd")

	original := fixtures.NewNoxuInstance("default", "foo")
	unstructured.SetNestedField(original.UnstructuredContent(), int64(42), "metadata", "unknown")
	unstructured.SetNestedField(original.UnstructuredContent(), nil, "metadata", "generation")

	unstructured.SetNestedField(original.UnstructuredContent(), map[string]interface{}{"foo": int64(42), "bar": "abc"}, "metadata", "annotations")
	unstructured.SetNestedField(original.UnstructuredContent(), map[string]interface{}{"invalid": "x y"}, "metadata", "labels")
	unstructured.SetNestedField(original.UnstructuredContent(), int64(42), "embedded", "metadata", "unknown")
	unstructured.SetNestedField(original.UnstructuredContent(), map[string]interface{}{"foo": int64(42), "bar": "abc"}, "embedded", "metadata", "annotations")
	unstructured.SetNestedField(original.UnstructuredContent(), map[string]interface{}{"invalid": "x y"}, "embedded", "metadata", "labels")
	unstructured.SetNestedField(original.UnstructuredContent(), "Foo", "embedded", "kind")
	unstructured.SetNestedField(original.UnstructuredContent(), "foo/v1", "embedded", "apiVersion")

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := path.Join("/", restOptions.StorageConfig.Prefix, noxuDefinition.Spec.Group, "noxus/default/foo")
	val, _ := json.Marshal(original.UnstructuredContent())
	if _, err := etcdclient.Put(ctx, key, string(val)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Checking that invalid objects can be deleted")
	noxuResourceClient := newNamespacedCustomResourceClient("default", dynamicClient, noxuDefinition)
	if err := noxuResourceClient.Delete(context.TODO(), "foo", metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Unexpected delete error %v", err)
	}
	if _, err := etcdclient.Put(ctx, key, string(val)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Checking that ObjectMeta is pruned from unknown fields")
	obj, err := noxuResourceClient.Get(context.TODO(), "foo", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	objJSON, _ := json.Marshal(obj.Object)
	t.Logf("Got object: %v", string(objJSON))

	if unknown, found, err := unstructured.NestedFieldNoCopy(obj.UnstructuredContent(), "metadata", "unknown"); err != nil {
		t.Errorf("Unexpected error: %v", err)
	} else if found {
		t.Errorf("Unexpected to find metadata.unknown=%#v", unknown)
	}
	if unknown, found, err := unstructured.NestedFieldNoCopy(obj.UnstructuredContent(), "embedded", "metadata", "unknown"); err != nil {
		t.Errorf("Unexpected error: %v", err)
	} else if found {
		t.Errorf("Unexpected to find embedded.metadata.unknown=%#v", unknown)
	}

	t.Logf("Checking that metadata.generation=1")

	if generation, found, err := unstructured.NestedInt64(obj.UnstructuredContent(), "metadata", "generation"); err != nil {
		t.Errorf("unexpected error getting metadata.generation: %v", err)
	} else if !found {
		t.Errorf("expected metadata.generation=1: got: %d", generation)
	} else if generation != 1 {
		t.Errorf("unexpected metadata.generation=%d: expected this to be set to 1", generation)
	}

	t.Logf("Checking that ObjectMeta is pruned from wrongly-typed annotations")

	if annotations, found, err := unstructured.NestedStringMap(obj.UnstructuredContent(), "metadata", "annotations"); err != nil {
		t.Errorf("Unexpected error: %v", err)
	} else if found {
		t.Errorf("Unexpected to find metadata.annotations: %#v", annotations)
	}
	if annotations, found, err := unstructured.NestedStringMap(obj.UnstructuredContent(), "embedded", "metadata", "annotations"); err != nil {
		t.Errorf("Unexpected error: %v", err)
	} else if found {
		t.Errorf("Unexpected to find embedded.metadata.annotations: %#v", annotations)
	}

	t.Logf("Checking that ObjectMeta still has the non-validating labels")

	if labels, found, err := unstructured.NestedStringMap(obj.UnstructuredContent(), "metadata", "labels"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if !found {
		t.Errorf("Expected to find metadata.labels, but didn't")
	} else if expected := map[string]string{"invalid": "x y"}; !reflect.DeepEqual(labels, expected) {
		t.Errorf("Expected metadata.labels to be %#v, got: %#v", expected, labels)
	}
	if labels, found, err := unstructured.NestedStringMap(obj.UnstructuredContent(), "embedded", "metadata", "labels"); err != nil {
		t.Errorf("Unexpected error: %v", err)
	} else if !found {
		t.Errorf("Expected to find embedded.metadata.labels, but didn't")
	} else if expected := map[string]string{"invalid": "x y"}; !reflect.DeepEqual(labels, expected) {
		t.Errorf("Expected embedded.metadata.labels to be %#v, got: %#v", expected, labels)
	}

	t.Logf("Trying to fail on updating with invalid labels")
	unstructured.SetNestedField(obj.Object, "changed", "metadata", "labels", "something")
	if got, err := noxuResourceClient.Update(context.TODO(), obj, metav1.UpdateOptions{}); err == nil {
		objJSON, _ := json.Marshal(obj.Object)
		gotJSON, _ := json.Marshal(got.Object)
		t.Fatalf("Expected update error, but didn't get one\nin: %s\nresponse: %v", string(objJSON), string(gotJSON))
	}

	t.Logf("Trying to fail on updating with invalid embedded label")
	unstructured.SetNestedField(obj.Object, "fixed", "metadata", "labels", "invalid")
	if got, err := noxuResourceClient.Update(context.TODO(), obj, metav1.UpdateOptions{}); err == nil {
		objJSON, _ := json.Marshal(obj.Object)
		gotJSON, _ := json.Marshal(got.Object)
		t.Fatalf("Expected update error, but didn't get one\nin: %s\nresponse: %v", string(objJSON), string(gotJSON))
	}

	t.Logf("Fixed all labels and update should work")
	unstructured.SetNestedField(obj.Object, "fixed", "embedded", "metadata", "labels", "invalid")
	if _, err := noxuResourceClient.Update(context.TODO(), obj, metav1.UpdateOptions{}); err != nil {
		t.Errorf("Unexpected update error with fixed labels: %v", err)
	}

	t.Logf("Trying to fail on updating with wrongly-typed embedded label")
	unstructured.SetNestedField(obj.Object, int64(42), "embedded", "metadata", "labels", "invalid")
	if got, err := noxuResourceClient.Update(context.TODO(), obj, metav1.UpdateOptions{}); err == nil {
		objJSON, _ := json.Marshal(obj.Object)
		gotJSON, _ := json.Marshal(got.Object)
		t.Fatalf("Expected update error, but didn't get one\nin: %s\nresponse: %v", string(objJSON), string(gotJSON))
	}
}

var embeddedResourceFixture = &apiextensionsv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{Name: "foos.tests.example.com"},
	Spec: apiextensionsv1.CustomResourceDefinitionSpec{
		Group: "tests.example.com",
		Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1beta1",
				Storage: true,
				Served:  true,
				Subresources: &apiextensionsv1.CustomResourceSubresources{
					Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
				},
			},
		},
		Names: apiextensionsv1.CustomResourceDefinitionNames{
			Plural:   "foos",
			Singular: "foo",
			Kind:     "Foo",
			ListKind: "FooList",
		},
		Scope:                 apiextensionsv1.ClusterScoped,
		PreserveUnknownFields: false,
	},
}

const (
	embeddedResourceSchema = `
type: object
properties:
  embedded:
    type: object
    x-kubernetes-embedded-resource: true
    x-kubernetes-preserve-unknown-fields: true
  noEmbeddedObject:
    type: object
    x-kubernetes-preserve-unknown-fields: true
  embeddedNested:
    type: object
    x-kubernetes-embedded-resource: true
    x-kubernetes-preserve-unknown-fields: true
    properties:
      embedded:
        type: object
        x-kubernetes-embedded-resource: true
        x-kubernetes-preserve-unknown-fields: true
  defaults:
    type: object
    x-kubernetes-embedded-resource: true
    x-kubernetes-preserve-unknown-fields: true
    default:
      apiVersion: v1
      kind: Pod
      labels:
        foo: bar
`

	embeddedResourceInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
embedded:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
    unspecified: bar
noEmbeddedObject:
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
    unspecified: bar
  embedded:
    apiVersion: foo/v1
    kind: Foo
    metadata:
      name: foo
      unspecified: bar
`

	expectedEmbeddedResourceInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
embedded:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: foo
noEmbeddedObject:
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
  embedded:
    apiVersion: foo/v1
    kind: Foo
    metadata:
      name: foo
defaults:
  apiVersion: v1
  kind: Pod
  labels:
    foo: bar
`

	wronglyTypedEmbeddedResourceInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
embedded:
  apiVersion: foo/v1
  kind: Foo
  metadata:
    name: instance
    namespace: 42
`

	invalidEmbeddedResourceInstance = `
kind: Foo
apiVersion: tests.example.com/v1beta1
embedded:
  apiVersion: foo/v1
  kind: "%"
  metadata:
    name: ..
embeddedNested:
  apiVersion: foo/v1
  kind: "%"
  metadata:
    name: ..
  embedded:
    apiVersion: foo/v1
    kind: "%"
    metadata:
      name: ..
invalidDefaults: {}
`
)

func TestEmbeddedResources(t *testing.T) {
	tearDownFn, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDownFn()

	crd := embeddedResourceFixture.DeepCopy()
	crd.Spec.Versions[0].Schema = &apiextensionsv1.CustomResourceValidation{}
	if err := yaml.Unmarshal([]byte(embeddedResourceSchema), &crd.Spec.Versions[0].Schema.OpenAPIV3Schema); err != nil {
		t.Fatal(err)
	}

	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'unspecified' fields to be pruned inside ObjectMetas")
	fooClient := dynamicClient.Resource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural})
	foo := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(embeddedResourceInstance), &foo.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(foo.Object, "foo", "metadata", "name")
	foo, err = fooClient.Create(context.TODO(), foo, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	t.Logf("CR created: %#v", foo.UnstructuredContent())

	t.Logf("Checking that everything unknown inside ObjectMeta is gone")
	delete(foo.Object, "metadata")
	var expected map[string]interface{}
	if err := yaml.Unmarshal([]byte(expectedEmbeddedResourceInstance), &expected); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, foo.Object) {
		t.Errorf("unexpected diff: %s", cmp.Diff(expected, foo.Object))
	}

	t.Logf("Trying to create wrongly typed CR")
	wronglyTyped := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(wronglyTypedEmbeddedResourceInstance), &wronglyTyped.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(wronglyTyped.Object, "invalid", "metadata", "name")
	_, err = fooClient.Create(context.TODO(), wronglyTyped, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("Expected creation to fail, but didn't")
	}
	t.Logf("Creation of wrongly typed object failed with: %v", err)

	for _, s := range []string{
		`embedded.metadata: Invalid value`,
	} {
		if !strings.Contains(err.Error(), s) {
			t.Errorf("missing error: %s", s)
		}
	}

	t.Logf("Trying to create invalid CR")
	invalid := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(invalidEmbeddedResourceInstance), &invalid.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(invalid.Object, "invalid", "metadata", "name")
	unstructured.SetNestedField(invalid.Object, "x y", "metadata", "labels", "foo")
	_, err = fooClient.Create(context.TODO(), invalid, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("Expected creation to fail, but didn't")
	}
	t.Logf("Creation of invalid object failed with: %v", err)

	invalidErrors := []string{
		`[metadata.labels: Invalid value: "x y"`,
		` embedded.kind: Invalid value: "%"`,
		` embedded.metadata.name: Invalid value: ".."`,
		` embeddedNested.kind: Invalid value: "%"`,
		` embeddedNested.metadata.name: Invalid value: ".."`,
		` embeddedNested.embedded.kind: Invalid value: "%"`,
		` embeddedNested.embedded.metadata.name: Invalid value: ".."`,
	}
	for _, s := range invalidErrors {
		if !strings.Contains(err.Error(), s) {
			t.Errorf("missing error: %s", s)
		}
	}

	t.Logf("Creating a valid CR and then updating it with invalid values, expecting the same errors")
	valid := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(embeddedResourceInstance), &valid.Object); err != nil {
		t.Fatal(err)
	}
	unstructured.SetNestedField(valid.Object, "valid", "metadata", "name")
	valid, err = fooClient.Create(context.TODO(), valid, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unable to create CR: %v", err)
	}
	for k, v := range invalid.Object {
		if k == "metadata" {
			continue
		}
		valid.Object[k] = v
	}
	unstructured.SetNestedField(valid.Object, "x y", "metadata", "labels", "foo")
	if _, err = fooClient.Update(context.TODO(), valid, metav1.UpdateOptions{}); err == nil {
		t.Fatal("Expected update error, but got none")
	}
	t.Logf("Update failed with: %v", err)
	for _, s := range invalidErrors {
		if !strings.Contains(err.Error(), s) {
			t.Errorf("missing error: %s", s)
		}
	}
}
