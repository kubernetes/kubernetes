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

package fake

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

// runtimeObject matches objects implementing runtime.Object, and tracks the underlying type
// (runtime.Object is a pointer type, but we also want access to the target type)
type runtimeObject[T any] interface {
	*T
	runtime.Object
}

// objectWithMeta matches objects implementing both runtime.Object and metav1.Object.
type objectWithMeta[T any] interface {
	runtimeObject[T]
	metav1.Object
}

// namedPointerObject matches pointers implementing GetName(); it is intended for use with apply declarative configurations.
type namedPointerObject[T any] interface {
	*T
	GetName() *string
}

// FakeClient represents a fake client
type FakeClient[T any, PT objectWithMeta[T]] struct {
	*testing.Fake
	untypedFakeClient
}

type untypedFakeClient struct {
	namespace string
	resource  schema.GroupVersionResource
	kind      schema.GroupVersionKind
}

// FakeClientWithList represents a fake client with support for lists.
type FakeClientWithList[T, L any, PT objectWithMeta[T], PL runtimeObject[L]] struct {
	*FakeClient[T, PT]
	alsoFakeLister[T, L, PT, PL]
}

// FakeClientWithApply represents a fake client with support for apply declarative configurations.
type FakeClientWithApply[T, C any, PT objectWithMeta[T], PC namedPointerObject[C]] struct {
	*FakeClient[T, PT]
	alsoFakeApplier[T, C, PT, PC]
}

// FakeClientWithListAndApply represents a fake client with support for lists and apply declarative configurations.
type FakeClientWithListAndApply[T, L, C any, PT objectWithMeta[T], PL runtimeObject[L], PC namedPointerObject[C]] struct {
	*FakeClient[T, PT]
	alsoFakeLister[T, L, PT, PL]
	alsoFakeApplier[T, C, PT, PC]
}

// Helper types for composition
type alsoFakeLister[T, L any, PT objectWithMeta[T], PL runtimeObject[L]] struct {
	client       *FakeClient[T, PT]
	copyListMeta func(PL, PL)
	getItems     func(PL) []T
	setItems     func(PL, []T)
}

type alsoFakeApplier[T, C any, PT objectWithMeta[T], PC namedPointerObject[C]] struct {
	client *FakeClient[T, PT]
}

type Option func(*untypedFakeClient)

// ForResource declares the resource on which the fake client operates.
func ForResource(resource schema.GroupVersionResource) Option {
	return func(c *untypedFakeClient) { c.resource = resource }
}

// ForKind declares the kind on which the fake client operates.
func ForKind(kind schema.GroupVersionKind) Option {
	return func(c *untypedFakeClient) { c.kind = kind }
}

// InNamespace specifies the namespace in which the fake client operates.
func InNamespace(namespace string) Option {
	return func(c *untypedFakeClient) { c.namespace = namespace }
}

type ListOption[T, L any, PT objectWithMeta[T], PL runtimeObject[L]] func(a *alsoFakeLister[T, L, PT, PL])

// WithListMetaCopier provides a function to copy list metadata from one list to another.
func WithListMetaCopier[T, L any, PT objectWithMeta[T], PL runtimeObject[L]](copier func(PL, PL)) ListOption[T, L, PT, PL] {
	return func(a *alsoFakeLister[T, L, PT, PL]) {
		a.copyListMeta = copier
	}
}

// WithItemsGetter provides a function to retrieve the slice of items from a list.
func WithItemsGetter[T, L any, PT objectWithMeta[T], PL runtimeObject[L]](getter func(PL) []T) ListOption[T, L, PT, PL] {
	return func(a *alsoFakeLister[T, L, PT, PL]) {
		a.getItems = getter
	}
}

// WithItemsSetter provides a function to store a slice of items in a list.
func WithItemsSetter[T, L any, PT objectWithMeta[T], PL runtimeObject[L]](setter func(PL, []T)) ListOption[T, L, PT, PL] {
	return func(a *alsoFakeLister[T, L, PT, PL]) {
		a.setItems = setter
	}
}

// NewFakeClient constructs a fake client, namespaced or not, with no support for lists or apply.
func NewFakeClient[T any, PT objectWithMeta[T]](
	fake *testing.Fake, opts ...any,
) *FakeClient[T, PT] {
	c := &FakeClient[T, PT]{Fake: fake}
	for _, option := range opts {
		if opt, ok := option.(Option); ok {
			opt(&c.untypedFakeClient)
		}
	}
	return c
}

// NewFakeClientWithList constructs a client with support for lists.
func NewFakeClientWithList[T, L any, PT objectWithMeta[T], PL runtimeObject[L]](
	fake *testing.Fake, opts ...any,
) *FakeClientWithList[T, L, PT, PL] {
	fakeClient := NewFakeClient[T, PT](fake, opts...)
	return &FakeClientWithList[T, L, PT, PL]{
		fakeClient,
		newFakeLister[T, L, PT, PL](fakeClient, opts...),
	}
}

func newFakeLister[T, L any, PT objectWithMeta[T], PL runtimeObject[L]](
	fakeClient *FakeClient[T, PT], opts ...any,
) alsoFakeLister[T, L, PT, PL] {
	l := alsoFakeLister[T, L, PT, PL]{client: fakeClient}
	for _, option := range opts {
		if opt, ok := option.(ListOption[T, L, PT, PL]); ok {
			opt(&l)
		}
	}
	return l
}

// NewFakeClientWithApply constructs a client with support for apply declarative configurations.
func NewFakeClientWithApply[T, C any, PT objectWithMeta[T], PC namedPointerObject[C]](
	fake *testing.Fake, opts ...any,
) *FakeClientWithApply[T, C, PT, PC] {
	fakeClient := NewFakeClient[T, PT](fake, opts...)
	return &FakeClientWithApply[T, C, PT, PC]{
		fakeClient,
		alsoFakeApplier[T, C, PT, PC]{fakeClient},
	}
}

// NewFakeClientWithListAndApply constructs a client with support for lists and applying declarative configurations.
func NewFakeClientWithListAndApply[T, L, C any, PT objectWithMeta[T], PL runtimeObject[L], PC namedPointerObject[C]](
	fake *testing.Fake, opts ...any,
) *FakeClientWithListAndApply[T, L, C, PT, PL, PC] {
	fakeClient := NewFakeClient[T, PT](fake, opts...)
	return &FakeClientWithListAndApply[T, L, C, PT, PL, PC]{
		fakeClient,
		newFakeLister[T, L, PT, PL](fakeClient, opts...),
		alsoFakeApplier[T, C, PT, PC]{fakeClient},
	}
}

// Get takes name of a resource, and returns the corresponding object, and an error if there is any.
func (c *FakeClient[T, PT]) Get(ctx context.Context, name string, opts metav1.GetOptions) (PT, error) {
	emptyResult := new(T)

	obj, err := c.Fake.
		Invokes(testing.NewGetActionWithOptions(c.resource, c.namespace, name, opts), PT(emptyResult))
	if obj == nil {
		return emptyResult, err
	}
	return obj.(PT), err
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func (l *alsoFakeLister[T, L, PT, PL]) List(ctx context.Context, opts metav1.ListOptions) (result PL, err error) {
	emptyResult := new(L)
	obj, err := l.client.Fake.
		Invokes(testing.NewListActionWithOptions(l.client.resource, l.client.kind, l.client.namespace, opts), PL(emptyResult))
	if obj == nil {
		return emptyResult, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		// Everything matches
		return obj.(PL), nil
	}
	list := new(L)
	l.copyListMeta(list, obj.(PL))
	var items []T
	for _, item := range l.getItems(obj.(PL)) {
		itemMeta, err := meta.Accessor(&item)
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
func (c *FakeClient[T, PT]) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.Fake.
		InvokesWatch(testing.NewWatchActionWithOptions(c.resource, c.namespace, opts))
}

// Create takes the representation of a resource and creates it.  Returns the server's representation of the resource, and an error, if there is any.
func (c *FakeClient[T, PT]) Create(ctx context.Context, resource PT, opts metav1.CreateOptions) (result PT, err error) {
	emptyResult := new(T)
	obj, err := c.Fake.
		Invokes(testing.NewCreateActionWithOptions(c.resource, c.namespace, resource, opts), PT(emptyResult))
	if obj == nil {
		return emptyResult, err
	}
	return obj.(PT), err
}

// Update takes the representation of a resource and updates it. Returns the server's representation of the resource, and an error, if there is any.
func (c *FakeClient[T, PT]) Update(ctx context.Context, resource PT, opts metav1.UpdateOptions) (result PT, err error) {
	emptyResult := new(T)
	obj, err := c.Fake.
		Invokes(testing.NewUpdateActionWithOptions(c.resource, c.namespace, resource, opts), PT(emptyResult))
	if obj == nil {
		return emptyResult, err
	}
	return obj.(PT), err
}

// UpdateStatus updates the resource's status and returns the updated resource.
func (c *FakeClient[T, PT]) UpdateStatus(ctx context.Context, resource PT, opts metav1.UpdateOptions) (result PT, err error) {
	emptyResult := new(T)
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceActionWithOptions(c.resource, "status", c.namespace, resource, opts), PT(emptyResult))

	if obj == nil {
		return emptyResult, err
	}
	return obj.(PT), err
}

// Delete deletes the resource matching the given name. Returns an error if one occurs.
func (c *FakeClient[T, PT]) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteActionWithOptions(c.resource, c.namespace, name, opts), PT(new(T)))
	return err
}

// DeleteWithResult takes name of the resource and deletes it. Returns an error if one occurs, or a metav1.APIResult on success.
func (c *FakeClient[T, PT]) DeleteWithResult(ctx context.Context, name string, opts metav1.DeleteOptions) (metav1.APIResult, error) {
	obj, err := c.Fake.
		Invokes(testing.NewDeleteActionWithOptions(c.resource, c.namespace, name, opts), &metav1.Status{})

	fakeResult := testing.FakeAPIResult{
		Obj: obj,
		Err: err,
	}
	if err != nil {
		if statusErr, ok := err.(interface{ Status() metav1.Status }); ok {
			fakeResult.Code = int(statusErr.Status().Code)
		}
	} else if obj != nil {
		fakeResult.Code = 200
		if status, ok := obj.(*metav1.Status); ok {
			fakeResult.Code = int(status.Code)
		}
	}
	return fakeResult, err
}

// DeleteCollection deletes a collection of objects.
func (l *alsoFakeLister[T, L, PT, PL]) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	_, err := l.client.Fake.
		Invokes(testing.NewDeleteCollectionActionWithOptions(l.client.resource, l.client.namespace, opts, listOpts), PL(new(L)))
	return err
}

// Patch applies the patch and returns the patched resource.
func (c *FakeClient[T, PT]) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (result PT, err error) {
	emptyResult := new(T)
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(c.resource, c.namespace, name, pt, data, opts, subresources...), PT(emptyResult))
	if obj == nil {
		return emptyResult, err
	}
	return obj.(PT), err
}

// Apply takes the given apply declarative configuration, applies it and returns the applied resource.
func (a *alsoFakeApplier[T, C, PT, PC]) Apply(ctx context.Context, configuration PC, opts metav1.ApplyOptions) (result PT, err error) {
	emptyResult := new(T)
	if configuration == nil {
		return emptyResult, fmt.Errorf("configuration provided to Apply must not be nil")
	}
	data, err := json.Marshal(configuration)
	if err != nil {
		return emptyResult, err
	}
	name := configuration.GetName()
	if name == nil {
		return emptyResult, fmt.Errorf("configuration.Name must be provided to Apply")
	}
	obj, err := a.client.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(a.client.resource, a.client.namespace, *name, types.ApplyPatchType, data, opts.ToPatchOptions()), PT(emptyResult))
	if obj == nil {
		return emptyResult, err
	}
	return obj.(PT), err
}

// ApplyStatus applies the given apply declarative configuration to the resource's status and returns the updated resource.
func (a *alsoFakeApplier[T, C, PT, PC]) ApplyStatus(ctx context.Context, configuration PC, opts metav1.ApplyOptions) (result PT, err error) {
	emptyResult := new(T)
	if configuration == nil {
		return emptyResult, fmt.Errorf("configuration provided to Apply must not be nil")
	}
	data, err := json.Marshal(configuration)
	if err != nil {
		return emptyResult, err
	}
	name := configuration.GetName()
	if name == nil {
		return emptyResult, fmt.Errorf("configuration.Name must be provided to Apply")
	}
	obj, err := a.client.Fake.
		Invokes(testing.NewPatchSubresourceActionWithOptions(a.client.resource, a.client.namespace, *name, types.ApplyPatchType, data, opts.ToPatchOptions(), "status"), PT(emptyResult))

	if obj == nil {
		return emptyResult, err
	}
	return obj.(PT), err
}

// Namespace returns the namespace in which the fake client operates.
func (c *FakeClient[T, PT]) Namespace() string {
	return c.namespace
}

// Kind returns the kind on which the fake client operates.
func (c *FakeClient[T, PT]) Kind() schema.GroupVersionKind {
	return c.kind
}

// Resource returns the resource on which the fake client operates.
func (c *FakeClient[T, PT]) Resource() schema.GroupVersionResource {
	return c.resource
}
