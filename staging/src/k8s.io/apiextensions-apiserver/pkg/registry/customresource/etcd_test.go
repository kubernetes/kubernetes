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

package customresource_test

import (
	"io"
	"reflect"
	"strings"
	"testing"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	registrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/client-go/discovery"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource/tableconvertor"
)

func newStorage(t *testing.T) (customresource.CustomResourceStorage, *etcdtesting.EtcdTestServer) {
	server, etcdStorage := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)
	etcdStorage.Codec = unstructuredJsonCodec{}
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage, Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "noxus"}

	parameterScheme := runtime.NewScheme()
	parameterScheme.AddUnversionedTypes(schema.GroupVersion{Group: "mygroup.example.com", Version: "v1beta1"},
		&metav1.ListOptions{},
		&metav1.ExportOptions{},
		&metav1.GetOptions{},
		&metav1.DeleteOptions{},
	)

	typer := apiserver.UnstructuredObjectTyper{
		Delegate:          parameterScheme,
		UnstructuredTyper: discovery.NewUnstructuredObjectTyper(nil),
	}

	kind := schema.GroupVersionKind{Group: "mygroup.example.com", Version: "v1beta1", Kind: "Noxu"}

	labelSelectorPath := ".status.labelSelector"
	scale := &apiextensions.CustomResourceSubresourceScale{
		SpecReplicasPath:   ".spec.replicas",
		StatusReplicasPath: ".status.replicas",
		LabelSelectorPath:  &labelSelectorPath,
	}

	status := &apiextensions.CustomResourceSubresourceStatus{}

	// TODO: identify how to pass printer specification from the CRD
	table, _ := tableconvertor.New(nil)

	storage := customresource.NewStorage(
		schema.GroupResource{Group: "mygroup.example.com", Resource: "noxus"},
		schema.GroupVersionKind{Group: "mygroup.example.com", Version: "v1beta1", Kind: "NoxuItemList"},
		customresource.NewStrategy(
			typer,
			true,
			kind,
			nil,
			nil,
			status,
			scale,
		),
		restOptions,
		[]string{"all"},
		table,
	)

	return storage, server
}

// createCustomResource is a helper function that returns a CustomResource with the updated resource version.
func createCustomResource(storage *customresource.REST, cr unstructured.Unstructured, t *testing.T) (unstructured.Unstructured, error) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), cr.GetNamespace())
	obj, err := storage.Create(ctx, &cr, rest.ValidateAllObjectFunc, false)
	if err != nil {
		t.Errorf("Failed to create CustomResource, %v", err)
	}
	newCR := obj.(*unstructured.Unstructured)
	return *newCR, nil
}

func validNewCustomResource() *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1beta1",
			"kind":       "Noxu",
			"metadata": map[string]interface{}{
				"namespace": "default",
				"name":      "foo",
			},
			"spec": map[string]interface{}{
				"replicas": int64(7),
			},
		},
	}
}

var validCustomResource = *validNewCustomResource()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()
	test := registrytest.New(t, storage.CustomResource.Store)
	cr := validNewCustomResource()
	cr.SetNamespace("")
	test.TestCreate(
		cr,
	)
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()
	test := registrytest.New(t, storage.CustomResource.Store)
	test.TestGet(validNewCustomResource())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()
	test := registrytest.New(t, storage.CustomResource.Store)
	test.TestList(validNewCustomResource())
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()
	test := registrytest.New(t, storage.CustomResource.Store)
	test.TestDelete(validNewCustomResource())
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	expected := []string{"all"}
	actual := storage.CustomResource.Categories()
	ok := reflect.DeepEqual(actual, expected)
	if !ok {
		t.Errorf("categories are not equal. expected = %v actual = %v", expected, actual)
	}
}

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/foo"
	validCustomResource := validNewCustomResource()
	if err := storage.CustomResource.Storage.Create(ctx, key, validCustomResource, nil, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gottenObj, err := storage.CustomResource.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	update := gottenObj.(*unstructured.Unstructured)
	updateContent := update.Object
	updateContent["status"] = map[string]interface{}{
		"replicas": int64(7),
	}

	if _, _, err := storage.Status.Update(ctx, update.GetName(), rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	obj, err := storage.CustomResource.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cr, ok := obj.(*unstructured.Unstructured)
	if !ok {
		t.Fatal("unexpected error: custom resource should be of type Unstructured")
	}
	content := cr.UnstructuredContent()

	spec := content["spec"].(map[string]interface{})
	status := content["status"].(map[string]interface{})

	if spec["replicas"].(int64) != 7 {
		t.Errorf("we expected .spec.replicas to not be updated but it was updated to %v", spec["replicas"].(int64))
	}
	if status["replicas"].(int64) != 7 {
		t.Errorf("we expected .status.replicas to be updated to %d but it was %v", 7, status["replicas"].(int64))
	}
}

func TestScaleGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	name := "foo"

	var cr unstructured.Unstructured
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/" + name
	if err := storage.CustomResource.Storage.Create(ctx, key, &validCustomResource, &cr, 0); err != nil {
		t.Fatalf("error setting new custom resource (key: %s) %v: %v", key, validCustomResource, err)
	}

	want := &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:              cr.GetName(),
			Namespace:         metav1.NamespaceDefault,
			UID:               cr.GetUID(),
			ResourceVersion:   cr.GetResourceVersion(),
			CreationTimestamp: cr.GetCreationTimestamp(),
		},
		Spec: autoscalingv1.ScaleSpec{
			Replicas: int32(7),
		},
	}

	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}

	got := obj.(*autoscalingv1.Scale)
	if !apiequality.Semantic.DeepEqual(got, want) {
		t.Errorf("unexpected scale: %s", diff.ObjectDiff(got, want))
	}
}

func TestScaleGetWithoutSpecReplicas(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	name := "foo"

	var cr unstructured.Unstructured
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/" + name
	withoutSpecReplicas := validCustomResource.DeepCopy()
	unstructured.RemoveNestedField(withoutSpecReplicas.Object, "spec", "replicas")
	if err := storage.CustomResource.Storage.Create(ctx, key, withoutSpecReplicas, &cr, 0); err != nil {
		t.Fatalf("error setting new custom resource (key: %s) %v: %v", key, withoutSpecReplicas, err)
	}

	_, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err == nil {
		t.Fatalf("error expected for %s", name)
	}
	if expected := `the spec replicas field ".spec.replicas" does not exist`; !strings.Contains(err.Error(), expected) {
		t.Fatalf("expected error string %q, got: %v", expected, err)
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	name := "foo"

	var cr unstructured.Unstructured
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/" + name
	if err := storage.CustomResource.Storage.Create(ctx, key, &validCustomResource, &cr, 0); err != nil {
		t.Fatalf("error setting new custom resource (key: %s) %v: %v", key, validCustomResource, err)
	}

	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale, ok := obj.(*autoscalingv1.Scale)
	if !ok {
		t.Fatalf("%v is not of the type autoscalingv1.Scale", scale)
	}

	replicas := 12
	update := autoscalingv1.Scale{
		ObjectMeta: scale.ObjectMeta,
		Spec: autoscalingv1.ScaleSpec{
			Replicas: int32(replicas),
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}

	obj, err = storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale = obj.(*autoscalingv1.Scale)
	if scale.Spec.Replicas != int32(replicas) {
		t.Errorf("wrong replicas count: expected: %d got: %d", replicas, scale.Spec.Replicas)
	}

	update.ResourceVersion = scale.ResourceVersion
	update.Spec.Replicas = 15

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil && !errors.IsConflict(err) {
		t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
	}
}

func TestScaleUpdateWithoutSpecReplicas(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	name := "foo"

	var cr unstructured.Unstructured
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/" + name
	withoutSpecReplicas := validCustomResource.DeepCopy()
	unstructured.RemoveNestedField(withoutSpecReplicas.Object, "spec", "replicas")
	if err := storage.CustomResource.Storage.Create(ctx, key, withoutSpecReplicas, &cr, 0); err != nil {
		t.Fatalf("error setting new custom resource (key: %s) %v: %v", key, withoutSpecReplicas, err)
	}

	replicas := 12
	update := autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			ResourceVersion: cr.GetResourceVersion(),
		},
		Spec: autoscalingv1.ScaleSpec{
			Replicas: int32(replicas),
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}

	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale := obj.(*autoscalingv1.Scale)
	if scale.Spec.Replicas != int32(replicas) {
		t.Errorf("wrong replicas count: expected: %d got: %d", replicas, scale.Spec.Replicas)
	}
}

type unstructuredJsonCodec struct{}

func (c unstructuredJsonCodec) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	obj := into.(*unstructured.Unstructured)
	err := obj.UnmarshalJSON(data)
	if err != nil {
		return nil, nil, err
	}
	gvk := obj.GroupVersionKind()
	return obj, &gvk, nil
}

func (c unstructuredJsonCodec) Encode(obj runtime.Object, w io.Writer) error {
	u := obj.(*unstructured.Unstructured)
	bs, err := u.MarshalJSON()
	if err != nil {
		return err
	}
	w.Write(bs)
	return nil
}
