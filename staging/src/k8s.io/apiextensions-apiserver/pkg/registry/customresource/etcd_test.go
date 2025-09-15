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
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metainternal "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	registrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"

	apiextensionsinternal "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apiextensions-apiserver/pkg/crdserverscheme"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource/tableconvertor"
)

func newStorage(t *testing.T) (customresource.CustomResourceStorage, *etcd3testing.EtcdTestServer) {
	server, etcdStorage := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	etcdStorage.Codec = unstructured.UnstructuredJSONScheme
	groupResource := schema.GroupResource{Group: "mygroup.example.com", Resource: "noxus"}
	restOptions := generic.RESTOptions{StorageConfig: etcdStorage.ForResource(groupResource), Decorator: generic.UndecoratedStorage, DeleteCollectionWorkers: 1, ResourcePrefix: "noxus"}

	parameterScheme := runtime.NewScheme()
	parameterScheme.AddUnversionedTypes(schema.GroupVersion{Group: "mygroup.example.com", Version: "v1beta1"},
		&metav1.ListOptions{},
		&metav1.GetOptions{},
		&metav1.DeleteOptions{},
	)

	typer := apiserver.UnstructuredObjectTyper{
		Delegate:          parameterScheme,
		UnstructuredTyper: crdserverscheme.NewUnstructuredObjectTyper(),
	}

	kind := schema.GroupVersionKind{Group: "mygroup.example.com", Version: "v1beta1", Kind: "Noxu"}

	labelSelectorPath := ".status.labelSelector"
	scale := &apiextensionsinternal.CustomResourceSubresourceScale{
		SpecReplicasPath:   ".spec.replicas",
		StatusReplicasPath: ".status.replicas",
		LabelSelectorPath:  &labelSelectorPath,
	}

	status := &apiextensionsinternal.CustomResourceSubresourceStatus{}

	headers := []apiextensionsv1.CustomResourceColumnDefinition{
		{Name: "Age", Type: "date", JSONPath: ".metadata.creationTimestamp"},
		{Name: "Replicas", Type: "integer", JSONPath: ".spec.replicas"},
		{Name: "Missing", Type: "string", JSONPath: ".spec.missing"},
		{Name: "Invalid", Type: "integer", JSONPath: ".spec.string"},
		{Name: "String", Type: "string", JSONPath: ".spec.string"},
		{Name: "StringFloat64", Type: "string", JSONPath: ".spec.float64"},
		{Name: "StringInt64", Type: "string", JSONPath: ".spec.replicas"},
		{Name: "StringBool", Type: "string", JSONPath: ".spec.bool"},
		{Name: "Float64", Type: "number", JSONPath: ".spec.float64"},
		{Name: "Bool", Type: "boolean", JSONPath: ".spec.bool"},
	}
	table, _ := tableconvertor.New(headers)

	storage, err := customresource.NewStorage(
		groupResource,
		groupResource,
		kind,
		schema.GroupVersionKind{Group: "mygroup.example.com", Version: "v1beta1", Kind: "NoxuItemList"},
		customresource.NewStrategy(
			typer,
			true,
			kind,
			nil,
			nil,
			nil,
			status,
			scale,
			nil,
		),
		restOptions,
		[]string{"all"},
		table,
		managedfields.ResourcePathMappings{},
	)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	return storage, server
}

// createCustomResource is a helper function that returns a CustomResource with the updated resource version.
func createCustomResource(storage *customresource.REST, cr unstructured.Unstructured, t *testing.T) (unstructured.Unstructured, error) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), cr.GetNamespace())
	obj, err := storage.Create(ctx, &cr, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
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
				"namespace":         "default",
				"name":              "foo",
				"creationTimestamp": time.Now().Add(-time.Hour*12 - 30*time.Minute).UTC().Format(time.RFC3339),
			},
			"spec": map[string]interface{}{
				"replicas":         int64(7),
				"string":           "string",
				"float64":          float64(3.1415926),
				"bool":             true,
				"stringList":       []interface{}{"foo", "bar"},
				"mixedList":        []interface{}{"foo", int64(42)},
				"nonPrimitiveList": []interface{}{"foo", []interface{}{int64(1), int64(2)}},
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

func TestGenerationNumber(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()
	modifiedRno := *validNewCustomResource()
	modifiedRno.SetGeneration(10)
	ctx := genericapirequest.NewDefaultContext()
	cr, err := createCustomResource(storage.CustomResource, modifiedRno, t)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdCR, err := storage.CustomResource.Get(ctx, cr.GetName(), &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedCR, _ := etcdCR.(*unstructured.Unstructured)

	// Generation initialization
	if storedCR.GetGeneration() != 1 {
		t.Fatalf("Unexpected generation number %v", storedCR.GetGeneration())
	}

	// Updates to spec should increment the generation number
	setSpecReplicas(storedCR, getSpecReplicas(storedCR)+1)
	if _, _, err := storage.CustomResource.Update(ctx, storedCR.GetName(), rest.DefaultUpdatedObjectInfo(storedCR), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdCR, err = storage.CustomResource.Get(ctx, cr.GetName(), &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedCR, _ = etcdCR.(*unstructured.Unstructured)
	if storedCR.GetGeneration() != 2 {
		t.Fatalf("Unexpected generation, spec: %v", storedCR.GetGeneration())
	}

	// Updates to status should not increment the generation number
	setStatusReplicas(storedCR, getStatusReplicas(storedCR)+1)
	if _, _, err := storage.CustomResource.Update(ctx, storedCR.GetName(), rest.DefaultUpdatedObjectInfo(storedCR), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	etcdCR, err = storage.CustomResource.Get(ctx, cr.GetName(), &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	storedCR, _ = etcdCR.(*unstructured.Unstructured)
	if storedCR.GetGeneration() != 2 {
		t.Fatalf("Unexpected generation, spec: %v", storedCR.GetGeneration())
	}

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

func TestColumns(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/foo"
	validCustomResource := validNewCustomResource()
	if err := storage.CustomResource.Storage.Create(ctx, key, validCustomResource, nil, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gottenList, err := storage.CustomResource.List(ctx, &metainternal.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tbl, err := storage.CustomResource.ConvertToTable(ctx, gottenList, &metav1.TableOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedColumns := []struct {
		Name, Type string
	}{
		{"Name", "string"},
		{"Age", "date"},
		{"Replicas", "integer"},
		{"Missing", "string"},
		{"Invalid", "integer"},
		{"String", "string"},
		{"StringFloat64", "string"},
		{"StringInt64", "string"},
		{"StringBool", "string"},
		{"Float64", "number"},
		{"Bool", "boolean"},
	}
	if len(tbl.ColumnDefinitions) != len(expectedColumns) {
		t.Fatalf("got %d columns, expected %d. Got: %+v", len(tbl.ColumnDefinitions), len(expectedColumns), tbl.ColumnDefinitions)
	}
	for i, d := range tbl.ColumnDefinitions {
		if d.Name != expectedColumns[i].Name {
			t.Errorf("got column %d name %q, expected %q", i, d.Name, expectedColumns[i].Name)
		}
		if d.Type != expectedColumns[i].Type {
			t.Errorf("got column %d type %q, expected %q", i, d.Type, expectedColumns[i].Type)
		}
	}

	expectedRows := [][]interface{}{
		{
			"foo",
			"12h",
			int64(7),
			nil,
			nil,
			"string",
			"3.1415926",
			"7",
			"true",
			float64(3.1415926),
			true,
		},
	}
	for i, r := range tbl.Rows {
		if !reflect.DeepEqual(r.Cells, expectedRows[i]) {
			t.Errorf("got row %d with cells %#v, expected %#v", i, r.Cells, expectedRows[i])
		}
	}
}

func TestStatusUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/foo"
	validCustomResource := validNewCustomResource()
	if err := storage.CustomResource.Storage.Create(ctx, key, validCustomResource, nil, 0, false); err != nil {
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

	if _, _, err := storage.Status.Update(ctx, update.GetName(), rest.DefaultUpdatedObjectInfo(update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
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
	if err := storage.CustomResource.Storage.Create(ctx, key, &validCustomResource, &cr, 0, false); err != nil {
		t.Fatalf("error setting new custom resource (key: %s) %v: %v", key, validCustomResource, err)
	}

	want := &autoscalingv1.Scale{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Scale",
			APIVersion: "autoscaling/v1",
		},
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
		t.Errorf("unexpected scale: %s", cmp.Diff(got, want))
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
	if err := storage.CustomResource.Storage.Create(ctx, key, withoutSpecReplicas, &cr, 0, false); err != nil {
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
	if err := storage.CustomResource.Storage.Create(ctx, key, &validCustomResource, &cr, 0, false); err != nil {
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

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
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

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil && !errors.IsConflict(err) {
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
	if err := storage.CustomResource.Storage.Create(ctx, key, withoutSpecReplicas, &cr, 0, false); err != nil {
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

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
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

func TestScaleUpdateWithoutResourceVersion(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	name := "foo"

	var cr unstructured.Unstructured
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/" + name
	if err := storage.CustomResource.Storage.Create(ctx, key, &validCustomResource, &cr, 0, false); err != nil {
		t.Fatalf("error setting new custom resource (key: %s) %v: %v", key, validCustomResource, err)
	}

	replicas := int32(8)
	update := autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: autoscalingv1.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}

	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale := obj.(*autoscalingv1.Scale)
	if scale.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count: expected: %d got: %d", replicas, scale.Spec.Replicas)
	}
}

func TestScaleUpdateWithoutResourceVersionWithConflicts(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	name := "foo"

	var cr unstructured.Unstructured
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/" + name
	if err := storage.CustomResource.Storage.Create(ctx, key, &validCustomResource, &cr, 0, false); err != nil {
		t.Fatalf("error setting new custom resource (key: %s) %v: %v", key, validCustomResource, err)
	}

	fetchObject := func(name string) (*unstructured.Unstructured, error) {
		gotObj, err := storage.CustomResource.Get(ctx, name, &metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("error fetching custom resource %s: %v", name, err)
		}
		return gotObj.(*unstructured.Unstructured), nil
	}

	applyPatch := func(labelName, labelValue string) rest.TransformFunc {
		return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
			o := currentObject.(metav1.Object)
			o.SetLabels(map[string]string{
				labelName: labelValue,
			})
			return currentObject, nil
		}
	}

	errs := make(chan error, 1)
	rounds := 100
	go func() {
		// continuously submits a patch that updates a label and verifies the label update was effective
		labelName := "timestamp"
		for i := 0; i < rounds; i++ {
			expectedLabelValue := fmt.Sprint(i)
			update, err := fetchObject(name)
			if err != nil {
				errs <- err
				return
			}
			setNestedField(update, expectedLabelValue, "metadata", "labels", labelName)
			if _, _, err := storage.CustomResource.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyPatch(labelName, fmt.Sprint(i))), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {

				errs <- fmt.Errorf("error updating custom resource label: %v", err)
				return
			}

			gotObj, err := fetchObject(name)
			if err != nil {
				errs <- err
				return
			}
			gotLabelValue, _, err := unstructured.NestedString(gotObj.Object, "metadata", "labels", labelName)
			if err != nil {
				errs <- fmt.Errorf("error getting label %s of custom resource %s: %v", labelName, name, err)
				return
			}
			if gotLabelValue != expectedLabelValue {
				errs <- fmt.Errorf("wrong label value: expected: %s, got: %s", expectedLabelValue, gotLabelValue)
				return
			}
		}
	}()

	replicas := int32(0)
	update := autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
	// continuously submits a scale update without a resourceVersion for a monotonically increasing replica value
	// and verifies the scale update was effective
	for i := 0; i < rounds; i++ {
		select {
		case err := <-errs:
			t.Fatal(err)
		default:
			replicas++
			update.Spec.Replicas = replicas
			if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
				t.Fatalf("error updating scale %v: %v", update, err)
			}

			obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error fetching scale for %s: %v", name, err)
			}
			scale := obj.(*autoscalingv1.Scale)
			if scale.Spec.Replicas != replicas {
				t.Errorf("wrong replicas count: expected: %d got: %d", replicas, scale.Spec.Replicas)
			}
		}
	}
}

func TestScaleUpdateWithResourceVersionWithConflicts(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.CustomResource.Store.DestroyFunc()

	name := "foo"

	var cr unstructured.Unstructured
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := "/noxus/" + metav1.NamespaceDefault + "/" + name
	if err := storage.CustomResource.Storage.Create(ctx, key, &validCustomResource, &cr, 0, false); err != nil {
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

	replicas := int32(12)
	update := autoscalingv1.Scale{
		ObjectMeta: scale.ObjectMeta,
		Spec: autoscalingv1.ScaleSpec{
			Replicas: replicas,
		},
	}
	update.ResourceVersion = "1"

	_, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err == nil {
		t.Fatal("expecting an update conflict error")
	}
	if !errors.IsConflict(err) {
		t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
	}
}

func setSpecReplicas(u *unstructured.Unstructured, replicas int64) {
	setNestedField(u, replicas, "spec", "replicas")
}

func getSpecReplicas(u *unstructured.Unstructured) int64 {
	val, found, err := unstructured.NestedInt64(u.Object, "spec", "replicas")
	if !found || err != nil {
		return 0
	}
	return val
}

func setStatusReplicas(u *unstructured.Unstructured, replicas int64) {
	setNestedField(u, replicas, "status", "replicas")
}

func getStatusReplicas(u *unstructured.Unstructured) int64 {
	val, found, err := unstructured.NestedInt64(u.Object, "status", "replicas")
	if !found || err != nil {
		return 0
	}
	return val
}

func setNestedField(u *unstructured.Unstructured, value interface{}, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	unstructured.SetNestedField(u.Object, value, fields...)
}
