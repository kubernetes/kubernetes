/*
Copyright 2024 The Kubernetes Authors.

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

package gentype

import (
	"context"
	json "encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeClient represents a fake client
type FakeClient[T objectWithMeta] struct {
	*testing.Fake
	ns        string
	resource  schema.GroupVersionResource
	kind      schema.GroupVersionKind
	newObject func() T
}

// FakeClientWithList represents a fake client with support for lists.
type FakeClientWithList[T objectWithMeta, L runtime.Object] struct {
	*FakeClient[T]
	alsoFakeLister[T, L]
}

// FakeClientWithApply represents a fake client with support for apply declarative configurations.
type FakeClientWithApply[T objectWithMeta, C namedObject] struct {
	*FakeClient[T]
	alsoFakeApplier[T, C]
}

// FakeClientWithListAndApply represents a fake client with support for lists and apply declarative configurations.
type FakeClientWithListAndApply[T objectWithMeta, L runtime.Object, C namedObject] struct {
	*FakeClient[T]
	alsoFakeLister[T, L]
	alsoFakeApplier[T, C]
}

// Helper types for composition
type alsoFakeLister[T objectWithMeta, L runtime.Object] struct {
	client       *FakeClient[T]
	newList      func() L
	copyListMeta func(L, L)
	getItems     func(L) []T
	setItems     func(L, []T)
}

type alsoFakeApplier[T objectWithMeta, C namedObject] struct {
	client *FakeClient[T]
}

// NewFakeClient constructs a fake client, namespaced or not, with no support for lists or apply.
// Non-namespaced clients are constructed by passing an empty namespace ("").
func NewFakeClient[T objectWithMeta](
	fake *testing.Fake, namespace string, resource schema.GroupVersionResource, kind schema.GroupVersionKind, emptyObjectCreator func() T,
) *FakeClient[T] {
	return &FakeClient[T]{fake, namespace, resource, kind, emptyObjectCreator}
}

// NewFakeClientWithList constructs a namespaced client with support for lists.
func NewFakeClientWithList[T objectWithMeta, L runtime.Object](
	fake *testing.Fake, namespace string, resource schema.GroupVersionResource, kind schema.GroupVersionKind, emptyObjectCreator func() T,
	emptyListCreator func() L, listMetaCopier func(L, L), itemGetter func(L) []T, itemSetter func(L, []T),
) *FakeClientWithList[T, L] {
	fakeClient := NewFakeClient[T](fake, namespace, resource, kind, emptyObjectCreator)
	return &FakeClientWithList[T, L]{
		fakeClient,
		alsoFakeLister[T, L]{fakeClient, emptyListCreator, listMetaCopier, itemGetter, itemSetter},
	}
}

// NewFakeClientWithApply constructs a namespaced client with support for apply declarative configurations.
func NewFakeClientWithApply[T objectWithMeta, C namedObject](
	fake *testing.Fake, namespace string, resource schema.GroupVersionResource, kind schema.GroupVersionKind, emptyObjectCreator func() T,
) *FakeClientWithApply[T, C] {
	fakeClient := NewFakeClient[T](fake, namespace, resource, kind, emptyObjectCreator)
	return &FakeClientWithApply[T, C]{
		fakeClient,
		alsoFakeApplier[T, C]{fakeClient},
	}
}

// NewFakeClientWithListAndApply constructs a client with support for lists and applying declarative configurations.
func NewFakeClientWithListAndApply[T objectWithMeta, L runtime.Object, C namedObject](
	fake *testing.Fake, namespace string, resource schema.GroupVersionResource, kind schema.GroupVersionKind, emptyObjectCreator func() T,
	emptyListCreator func() L, listMetaCopier func(L, L), itemGetter func(L) []T, itemSetter func(L, []T),
) *FakeClientWithListAndApply[T, L, C] {
	fakeClient := NewFakeClient[T](fake, namespace, resource, kind, emptyObjectCreator)
	return &FakeClientWithListAndApply[T, L, C]{
		fakeClient,
		alsoFakeLister[T, L]{fakeClient, emptyListCreator, listMetaCopier, itemGetter, itemSetter},
		alsoFakeApplier[T, C]{fakeClient},
	}
}

// Get takes name of a resource, and returns the corresponding object, and an error if there is any.
func (c *FakeClient[T]) Get(ctx context.Context, name string, options metav1.GetOptions) (T, error) {
	emptyResult := c.newObject()

	obj, err := c.Fake.
		Invokes(testing.NewGetActionWithOptions(c.resource, c.ns, name, options), emptyResult)
	if obj == nil {
		return emptyResult, err
	}
	return obj.(T), err
}

func ToPointerSlice[T any](src []T) []*T {
	if src == nil {
		return nil
	}
	result := make([]*T, len(src))
	for i := range src {
		result[i] = &src[i]
	}
	return result
}

func FromPointerSlice[T any](src []*T) []T {
	if src == nil {
		return nil
	}
	result := make([]T, len(src))
	for i := range src {
		result[i] = *src[i]
	}
	return result
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func (l *alsoFakeLister[T, L]) List(ctx context.Context, opts metav1.ListOptions) (result L, err error) {
	emptyResult := l.newList()
	obj, err := l.client.Fake.
		Invokes(testing.NewListActionWithOptions(l.client.resource, l.client.kind, l.client.ns, opts), emptyResult)
	if obj == nil {
		return emptyResult, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		// Everything matches
		return obj.(L), nil
	}
	list := l.newList()
	l.copyListMeta(list, obj.(L))
	var items []T
	for _, item := range l.getItems(obj.(L)) {
		itemMeta, err := meta.Accessor(item)
		if err != nil {
			// No ObjectMeta, nothing can match
			continue
		}
		if label.Matches(labels.Set(itemMeta.GetLabels())) {
			items = append(items, item)
		}
	}
	l.setItems(list, items)
	return list, err
}

// Watch returns a watch.Interface that watches the requested resources.
func (c *FakeClient[T]) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchActionWithOptions(c.resource, c.ns, opts))
}

// Create takes the representation of a resource and creates it.  Returns the server's representation of the resource, and an error, if there is any.
func (c *FakeClient[T]) Create(ctx context.Context, resource T, opts metav1.CreateOptions) (result T, err error) {
	emptyResult := c.newObject()
	obj, err := c.Fake.
		Invokes(testing.NewCreateActionWithOptions(c.resource, c.ns, resource, opts), emptyResult)
	if obj == nil {
		return emptyResult, err
	}
	return obj.(T), err
}

// Update takes the representation of a resource and updates it. Returns the server's representation of the resource, and an error, if there is any.
func (c *FakeClient[T]) Update(ctx context.Context, resource T, opts metav1.UpdateOptions) (result T, err error) {
	emptyResult := c.newObject()
	obj, err := c.Fake.
		Invokes(testing.NewUpdateActionWithOptions(c.resource, c.ns, resource, opts), emptyResult)
	if obj == nil {
		return emptyResult, err
	}
	return obj.(T), err
}

// UpdateStatus updates the resource's status and returns the updated resource.
func (c *FakeClient[T]) UpdateStatus(ctx context.Context, resource T, opts metav1.UpdateOptions) (result T, err error) {
	emptyResult := c.newObject()
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceActionWithOptions(c.resource, "status", c.ns, resource, opts), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(T), err
}

// Delete deletes the resource matching the given name. Returns an error if one occurs.
func (c *FakeClient[T]) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteActionWithOptions(c.resource, c.ns, name, opts), c.newObject())
	return err
}

// DeleteCollection deletes a collection of objects.
func (l *alsoFakeLister[T, L]) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	_, err := l.client.Fake.
		Invokes(testing.NewDeleteCollectionActionWithOptions(l.client.resource, l.client.ns, opts, listOpts), l.newList())
	return err
}

// Patch applies the patch and returns the patched resource.
func (c *FakeClient[T]) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (result T, err error) {
	emptyResult := c.newObject()
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(c.resource, c.ns, name, pt, data, opts, subresources...), emptyResult)
	if obj == nil {
		return emptyResult, err
	}
	return obj.(T), err
}

// Apply takes the given apply declarative configuration, applies it and returns the applied resource.
func (a *alsoFakeApplier[T, C]) Apply(ctx context.Context, configuration C, opts metav1.ApplyOptions) (result T, err error) {
	if configuration == *new(C) {
		return *new(T), fmt.Errorf("configuration provided to Apply must not be nil")
	}
	data, err := json.Marshal(configuration)
	if err != nil {
		return *new(T), err
	}
	name := configuration.GetName()
	if name == nil {
		return *new(T), fmt.Errorf("configuration.Name must be provided to Apply")
	}
	emptyResult := a.client.newObject()
	obj, err := a.client.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(a.client.resource, a.client.ns, *name, types.ApplyPatchType, data, opts.ToPatchOptions()), emptyResult)
	if obj == nil {
		return emptyResult, err
	}
	return obj.(T), err
}

// ApplyStatus applies the given apply declarative configuration to the resource's status and returns the updated resource.
func (a *alsoFakeApplier[T, C]) ApplyStatus(ctx context.Context, configuration C, opts metav1.ApplyOptions) (result T, err error) {
	if configuration == *new(C) {
		return *new(T), fmt.Errorf("configuration provided to Apply must not be nil")
	}
	data, err := json.Marshal(configuration)
	if err != nil {
		return *new(T), err
	}
	name := configuration.GetName()
	if name == nil {
		return *new(T), fmt.Errorf("configuration.Name must be provided to Apply")
	}
	emptyResult := a.client.newObject()
	obj, err := a.client.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(a.client.resource, a.client.ns, *name, types.ApplyPatchType, data, opts.ToPatchOptions(), "status"), emptyResult)

	if obj == nil {
		return emptyResult, err
	}
	return obj.(T), err
}

func (c *FakeClient[T]) Namespace() string {
	return c.ns
}

func (c *FakeClient[T]) Kind() schema.GroupVersionKind {
	return c.kind
}

func (c *FakeClient[T]) Resource() schema.GroupVersionResource {
	return c.resource
}
