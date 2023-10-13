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

package generic

import (
	"context"
	json "encoding/json"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// objectWithMeta matches objects implementing both runtime.Object and metav1.Object.
type objectWithMeta interface {
	runtime.Object
	metav1.Object
}

// namedObject matches comparable objects implementing GetName(); it is intended for use with apply declarative configurations.
type namedObject interface {
	comparable
	GetName() *string
}

// TypeClient represents a client, optionally namespaced, with no support for lists or apply declarative configurations.
type TypeClient[T objectWithMeta] struct {
	Resource       string
	Client         rest.Interface
	Namespace      string
	newObject      func() T
	parameterCodec runtime.ParameterCodec
}

// TypeClientWithList represents a client with support for lists.
type TypeClientWithList[T objectWithMeta, L runtime.Object] struct {
	TypeClient[T]
	newList func() L
}

// TypeClientWithApply represents a client with support for apply declarative configurations.
type TypeClientWithApply[T objectWithMeta, C namedObject] struct {
	TypeClient[T]
}

// TypeClientWithListAndApply represents a client with support for lists and apply declarative configurations.
// It can't use both TypeClientWithList and TypeClientWithApply, because of the resulting ambiguous inclusion of TypeClient.
// Here it uses TypeClientWithApply and adds newList(); this means that list-related methods must be defined twice,
// once for TypeClientWithList and once for TypeClientWithListAndApply.
type TypeClientWithListAndApply[T objectWithMeta, L runtime.Object, C namedObject] struct {
	TypeClientWithApply[T, C]
	newList func() L
}

// NewNamespaced constructs a namespaced client.
func NewNamespaced[T objectWithMeta](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
) *TypeClient[T] {
	return &TypeClient[T]{
		Resource:       resource,
		Client:         client,
		parameterCodec: parameterCodec,
		Namespace:      namespace,
		newObject:      emptyObjectCreator,
	}
}

// NewNamespacedWithList constructs a namespaced client with support for lists.
func NewNamespacedWithList[T objectWithMeta, L runtime.Object](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
	emptyListCreator func() L,
) *TypeClientWithList[T, L] {
	return &TypeClientWithList[T, L]{
		*NewNamespaced[T](resource, client, parameterCodec, namespace, emptyObjectCreator),
		emptyListCreator,
	}
}

// NewNamespacedWithApply constructs a namespaced client with support for apply declarative configurations.
func NewNamespacedWithApply[T objectWithMeta, C namedObject](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
) *TypeClientWithApply[T, C] {
	return &TypeClientWithApply[T, C]{
		*NewNamespaced[T](resource, client, parameterCodec, namespace, emptyObjectCreator),
	}
}

// NewNamespacedWithListAndApply constructs a namespaced client with support for lists and apply declarative configurations.
func NewNamespacedWithListAndApply[T objectWithMeta, L runtime.Object, C namedObject](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
	emptyListCreator func() L,
) *TypeClientWithListAndApply[T, L, C] {
	return &TypeClientWithListAndApply[T, L, C]{
		*NewNamespacedWithApply[T, C](resource, client, parameterCodec, namespace, emptyObjectCreator),
		emptyListCreator,
	}
}

// NewNonNamespaced constructs a non-namespaced client.
func NewNonNamespaced[T objectWithMeta](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, emptyObjectCreator func() T,
) *TypeClient[T] {
	return &TypeClient[T]{
		Resource:       resource,
		Client:         client,
		parameterCodec: parameterCodec,
		newObject:      emptyObjectCreator,
	}
}

// NewNonNamespacedWithList constructs a non-namespaced client with support for lists.
func NewNonNamespacedWithList[T objectWithMeta, L runtime.Object](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, emptyObjectCreator func() T, emptyListCreator func() L,
) *TypeClientWithList[T, L] {
	return &TypeClientWithList[T, L]{
		*NewNonNamespaced[T](resource, client, parameterCodec, emptyObjectCreator),
		emptyListCreator,
	}
}

// NewNonNamespacedWithApply constructs a non-namespaced client with support for apply declarative configurations.
func NewNonNamespacedWithApply[T objectWithMeta, C namedObject](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, emptyObjectCreator func() T,
) *TypeClientWithApply[T, C] {
	return &TypeClientWithApply[T, C]{
		*NewNonNamespaced[T](resource, client, parameterCodec, emptyObjectCreator),
	}
}

// NewNonNamespacedWithListAndApply constructs a non-namespaced client with support for lists and apply declarative configurations.
func NewNonNamespacedWithListAndApply[T objectWithMeta, L runtime.Object, C namedObject](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, emptyObjectCreator func() T, emptyListCreator func() L,
) *TypeClientWithListAndApply[T, L, C] {
	return &TypeClientWithListAndApply[T, L, C]{
		*NewNonNamespacedWithApply[T, C](resource, client, parameterCodec, emptyObjectCreator),
		emptyListCreator,
	}
}

// Get takes name of the resource, and returns the corresponding object, and an error if there is any.
func (c *TypeClient[T]) Get(ctx context.Context, name string, options metav1.GetOptions) (T, error) {
	result := c.newObject()
	err := c.Client.Get().
		NamespaceIfScoped(c.Namespace, c.Namespace != "").
		Resource(c.Resource).
		Name(name).
		VersionedParams(&options, c.parameterCodec).
		Do(ctx).
		Into(result)
	return result, err
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func (l *TypeClientWithList[T, L]) List(ctx context.Context, opts metav1.ListOptions) (L, error) {
	return list(ctx, l.TypeClient, l.newList, l.parameterCodec, opts)
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func (l *TypeClientWithListAndApply[T, L, C]) List(ctx context.Context, opts metav1.ListOptions) (L, error) {
	return list(ctx, l.TypeClient, l.newList, l.parameterCodec, opts)
}

func list[T objectWithMeta, L runtime.Object](
	ctx context.Context, client TypeClient[T], newList func() L, parameterCodec runtime.ParameterCodec, opts metav1.ListOptions,
) (L, error) {
	list := newList()
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	err := client.Client.Get().
		NamespaceIfScoped(client.Namespace, client.Namespace != "").
		Resource(client.Resource).
		VersionedParams(&opts, parameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(list)
	return list, err
}

// Watch returns a watch.Interface that watches the requested resources.
func (c *TypeClient[T]) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.Client.Get().
		NamespaceIfScoped(c.Namespace, c.Namespace != "").
		Resource(c.Resource).
		VersionedParams(&opts, c.parameterCodec).
		Timeout(timeout).
		Watch(ctx)
}

// Create takes the representation of a resource and creates it.  Returns the server's representation of the resource, and an error, if there is any.
func (c *TypeClient[T]) Create(ctx context.Context, obj T, opts metav1.CreateOptions) (T, error) {
	result := c.newObject()
	err := c.Client.Post().
		NamespaceIfScoped(c.Namespace, c.Namespace != "").
		Resource(c.Resource).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(result)
	return result, err
}

// Update takes the representation of a resource and updates it. Returns the server's representation of the resource, and an error, if there is any.
func (c *TypeClient[T]) Update(ctx context.Context, obj T, opts metav1.UpdateOptions) (T, error) {
	result := c.newObject()
	err := c.Client.Put().
		NamespaceIfScoped(c.Namespace, c.Namespace != "").
		Resource(c.Resource).
		Name(obj.GetName()).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(result)
	return result, err
}

// UpdateStatus updates the status subresource of a resource. Returns the server's representation of the resource, and an error, if there is any.
func (c *TypeClient[T]) UpdateStatus(ctx context.Context, obj T, opts metav1.UpdateOptions) (T, error) {
	result := c.newObject()
	err := c.Client.Put().
		NamespaceIfScoped(c.Namespace, c.Namespace != "").
		Resource(c.Resource).
		Name(obj.GetName()).
		SubResource("status").
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(result)
	return result, err
}

// Delete takes name of the resource and deletes it. Returns an error if one occurs.
func (c *TypeClient[T]) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	return c.Client.Delete().
		NamespaceIfScoped(c.Namespace, c.Namespace != "").
		Resource(c.Resource).
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}

// DeleteCollection deletes a collection of objects.
func (l *TypeClientWithList[T, L]) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	return deleteCollection[T](ctx, l.TypeClient, l.parameterCodec, opts, listOpts)
}

// DeleteCollection deletes a collection of objects.
func (l *TypeClientWithListAndApply[T, L, C]) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	return deleteCollection[T](ctx, l.TypeClient, l.parameterCodec, opts, listOpts)
}

func deleteCollection[T objectWithMeta](
	ctx context.Context, client TypeClient[T], parameterCodec runtime.ParameterCodec, opts metav1.DeleteOptions, listOpts metav1.ListOptions,
) error {
	var timeout time.Duration
	if listOpts.TimeoutSeconds != nil {
		timeout = time.Duration(*listOpts.TimeoutSeconds) * time.Second
	}
	return client.Client.Delete().
		NamespaceIfScoped(client.Namespace, client.Namespace != "").
		Resource(client.Resource).
		VersionedParams(&listOpts, parameterCodec).
		Timeout(timeout).
		Body(&opts).
		Do(ctx).
		Error()
}

// Patch applies the patch and returns the patched resource.
func (c *TypeClient[T]) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (T, error) {
	result := c.newObject()
	err := c.Client.Patch(pt).
		NamespaceIfScoped(c.Namespace, c.Namespace != "").
		Resource(c.Resource).
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, c.parameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return result, err
}

// Apply takes the given apply declarative configuration, applies it and returns the applied resource.
func (a *TypeClientWithApply[T, C]) Apply(ctx context.Context, obj C, opts metav1.ApplyOptions) (T, error) {
	result := a.newObject()
	if obj == *new(C) {
		return *new(T), fmt.Errorf("object provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	data, err := json.Marshal(obj)
	if err != nil {
		return *new(T), err
	}
	if obj.GetName() == nil {
		return *new(T), fmt.Errorf("obj.Name must be provided to Apply")
	}
	err = a.Client.Patch(types.ApplyPatchType).
		NamespaceIfScoped(a.Namespace, a.Namespace != "").
		Resource(a.Resource).
		Name(*obj.GetName()).
		VersionedParams(&patchOpts, a.parameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return result, err
}

// Apply takes the given apply declarative configuration, applies it to the status subresource and returns the applied resource.
func (a *TypeClientWithApply[T, C]) ApplyStatus(ctx context.Context, obj C, opts metav1.ApplyOptions) (T, error) {
	if obj == *new(C) {
		return *new(T), fmt.Errorf("object provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	data, err := json.Marshal(obj)
	if err != nil {
		return *new(T), err
	}

	if obj.GetName() == nil {
		return *new(T), fmt.Errorf("obj.Name must be provided to Apply")
	}

	result := a.newObject()
	err = a.Client.Patch(types.ApplyPatchType).
		NamespaceIfScoped(a.Namespace, a.Namespace != "").
		Resource(a.Resource).
		Name(*obj.GetName()).
		SubResource("status").
		VersionedParams(&patchOpts, a.parameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return result, err
}
