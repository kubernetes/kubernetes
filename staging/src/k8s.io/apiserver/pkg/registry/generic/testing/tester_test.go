/*
Copyright The Kubernetes Authors.

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

package tester

import (
	"context"
	"fmt"
	"path"
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
)

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

type testRESTStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	namespaceScoped          bool
	allowCreateOnUpdate      bool
	allowUnconditionalUpdate bool
}

func (t *testRESTStrategy) NamespaceScoped() bool { return t.namespaceScoped }
func (t *testRESTStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return t.allowCreateOnUpdate
}
func (t *testRESTStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return t.allowUnconditionalUpdate
}
func (t *testRESTStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {}
func (t *testRESTStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}
func (t *testRESTStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return nil
}
func (t *testRESTStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return nil
}
func (t *testRESTStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}
func (t *testRESTStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
func (t *testRESTStrategy) Canonicalize(obj runtime.Object) {}

// podDefaulter defaults the node name of pods without one, mirroring the
// defaulting that the API server would apply through the normal create path.
type podDefaulter struct{}

func (d *podDefaulter) Default(obj runtime.Object) {
	if pod, ok := obj.(*example.Pod); ok {
		if pod.Spec.NodeName == "" {
			pod.Spec.NodeName = "machine"
		}
	}
}

func newTestStore(t *testing.T) (factory.DestroyFunc, *genericregistry.Store) {
	podPrefix := "/pods/"
	server, sc := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	strategy := &testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}

	newFunc := func() runtime.Object { return &example.Pod{} }
	newListFunc := func() runtime.Object { return &example.PodList{} }

	sc.Codec = apitesting.TestStorageCodec(codecs, examplev1.SchemeGroupVersion)
	s, dFunc, err := factory.Create(*sc.ForResource(schema.GroupResource{Resource: "pods"}), newFunc, newListFunc, "/pods")
	if err != nil {
		t.Fatalf("Error creating storage: %v", err)
	}
	destroyFunc := func() {
		dFunc()
		server.Terminate(t)
	}

	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &example.Pod{} },
		NewListFunc:               func() runtime.Object { return &example.PodList{} },
		DefaultQualifiedResource:  example.Resource("pods"),
		SingularQualifiedResource: example.Resource("pod"),
		CreateStrategy:            strategy,
		UpdateStrategy:            strategy,
		DeleteStrategy:            strategy,
		KeyRootFunc: func(ctx context.Context) string {
			return podPrefix
		},
		KeyFunc: func(ctx context.Context, id string) (string, error) {
			if _, ok := genericapirequest.NamespaceFrom(ctx); !ok {
				return "", fmt.Errorf("namespace is required")
			}
			return path.Join(podPrefix, id), nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) { return obj.(*example.Pod).Name, nil },
		PredicateFunc: func(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
			return storage.SelectionPredicate{
				Label: label,
				Field: field,
				GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
					pod, ok := obj.(*example.Pod)
					if !ok {
						return nil, nil, fmt.Errorf("not a pod")
					}
					return labels.Set(pod.ObjectMeta.Labels), generic.ObjectMetaFieldsSet(&pod.ObjectMeta, true), nil
				},
			}
		},
		Storage: genericregistry.DryRunnableStorage{Storage: s},
	}
	return destroyFunc, store
}

func TestWithDefaulter(t *testing.T) {
	destroyFunc, store := newTestStore(t)
	defer destroyFunc()

	tester := New(t, store)
	defaulter := &podDefaulter{}

	if got := tester.WithDefaulter(defaulter); got != tester {
		t.Fatal("expected WithDefaulter to return the same Tester instance for chaining")
	}
	if tester.defaulter != defaulter {
		t.Fatal("expected defaulter to be set")
	}
}

func TestCreateObjectAppliesDefaults(t *testing.T) {
	destroyFunc, store := newTestStore(t)
	defer destroyFunc()

	tester := New(t, store).WithDefaulter(&podDefaulter{})
	ctx := tester.tester.TestContext()

	pod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test"},
	}
	if err := tester.createObject(ctx, pod); err != nil {
		t.Fatalf("unexpected error creating object: %v", err)
	}

	got, err := tester.getObject(ctx, pod)
	if err != nil {
		t.Fatalf("unexpected error getting object: %v", err)
	}
	stored := got.(*example.Pod)
	if stored.Spec.NodeName != "machine" {
		t.Errorf("expected default NodeName %q, got %q", "machine", stored.Spec.NodeName)
	}
}

func TestCreateObjectWithoutDefaulterDoesNotApplyDefaults(t *testing.T) {
	destroyFunc, store := newTestStore(t)
	defer destroyFunc()

	tester := New(t, store)
	ctx := tester.tester.TestContext()

	pod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test"},
	}
	if err := tester.createObject(ctx, pod); err != nil {
		t.Fatalf("unexpected error creating object: %v", err)
	}

	got, err := tester.getObject(ctx, pod)
	if err != nil {
		t.Fatalf("unexpected error getting object: %v", err)
	}
	stored := got.(*example.Pod)
	if stored.Spec.NodeName != "" {
		t.Errorf("expected empty NodeName without a defaulter, got %q", stored.Spec.NodeName)
	}
}

func TestSetObjectsForListAppliesDefaults(t *testing.T) {
	destroyFunc, store := newTestStore(t)
	defer destroyFunc()

	tester := New(t, store).WithDefaulter(&podDefaulter{})

	objects := []runtime.Object{
		&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test"}},
		&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: "test"}},
	}
	if returned := tester.setObjectsForList(objects); returned == nil {
		t.Fatal("expected setObjectsForList to return the objects")
	}

	for _, obj := range objects {
		pod := obj.(*example.Pod)
		got, err := tester.getObject(tester.tester.TestContext(), pod)
		if err != nil {
			t.Fatalf("unexpected error getting object %q: %v", pod.Name, err)
		}
		stored := got.(*example.Pod)
		if stored.Spec.NodeName != "machine" {
			t.Errorf("object %q: expected default NodeName %q, got %q", pod.Name, "machine", stored.Spec.NodeName)
		}
	}
}

func TestSetObjectsForListWithoutDefaulterDoesNotApplyDefaults(t *testing.T) {
	destroyFunc, store := newTestStore(t)
	defer destroyFunc()

	tester := New(t, store)

	objects := []runtime.Object{
		&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test"}},
	}
	if returned := tester.setObjectsForList(objects); returned == nil {
		t.Fatal("expected setObjectsForList to return the objects")
	}

	pod := objects[0].(*example.Pod)
	got, err := tester.getObject(tester.tester.TestContext(), pod)
	if err != nil {
		t.Fatalf("unexpected error getting object %q: %v", pod.Name, err)
	}
	stored := got.(*example.Pod)
	if stored.Spec.NodeName != "" {
		t.Errorf("expected empty NodeName without a defaulter, got %q", stored.Spec.NodeName)
	}
}
