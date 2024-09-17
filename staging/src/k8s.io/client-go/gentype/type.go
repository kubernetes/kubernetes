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
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	"k8s.io/client-go/util/apply"
	"k8s.io/client-go/util/consistencydetector"
	"k8s.io/client-go/util/watchlist"
	"k8s.io/klog/v2"
)

// objectWithMeta matches objects implementing both runtime.Object and metav1.Object.
type objectWithMeta interface {
	runtime.Object
	metav1.Object
}

type resource interface{}

// namedObject matches comparable objects implementing GetName(); it is intended for use with apply declarative configurations.
type namedObject interface {
	comparable
	GetName() *string
}

// Client represents a client, optionally namespaced, with no support for lists or apply declarative configurations.
type Client[T objectWithMeta] struct {
	ResourceClient
	newObject func() T
}

// Client represents a resource client client, optionally namespaced.
type ResourceClient struct {
	resource        string
	client          rest.Interface
	namespace       string // "" for non-namespaced clients
	parameterCodec  runtime.ParameterCodec
	prefersProtobuf bool
}

func resourceAsObject(r resource) runtime.Object {
	return any(&r).(runtime.Object)
}

// ClientWithList represents a client with support for lists.
type ClientWithList[T objectWithMeta, L runtime.Object] struct {
	*Client[T]
	alsoLister[T, L]
}

// ClientWithApply represents a client with support for apply declarative configurations.
type ClientWithApply[T objectWithMeta, C namedObject] struct {
	*Client[T]
	alsoApplier[T, C]
}

// ClientWithListAndApply represents a client with support for lists and apply declarative configurations.
type ClientWithListAndApply[T objectWithMeta, L runtime.Object, C namedObject] struct {
	*Client[T]
	alsoLister[T, L]
	alsoApplier[T, C]
}

// Helper types for composition
type alsoLister[T objectWithMeta, L runtime.Object] struct {
	client  *Client[T]
	newList func() L
}

type alsoApplier[T objectWithMeta, C namedObject] struct {
	client *Client[T]
}

type Option[T objectWithMeta] func(*Client[T])

func PrefersProtobuf[T objectWithMeta]() Option[T] {
	return func(c *Client[T]) { c.prefersProtobuf = true }
}

// NewClient constructs a client, namespaced or not, with no support for lists or apply.
// Non-namespaced clients are constructed by passing an empty namespace ("").
func NewClient[T objectWithMeta](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
	options ...Option[T],
) *Client[T] {
	c := &Client[T]{
		ResourceClient{
			resource:       resource,
			client:         client,
			parameterCodec: parameterCodec,
			namespace:      namespace,
		},
		emptyObjectCreator,
	}
	for _, option := range options {
		option(c)
	}
	return c
}

// NewClientWithList constructs a namespaced client with support for lists.
func NewClientWithList[T objectWithMeta, L runtime.Object](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
	emptyListCreator func() L, options ...Option[T],
) *ClientWithList[T, L] {
	typeClient := NewClient[T](resource, client, parameterCodec, namespace, emptyObjectCreator, options...)
	return &ClientWithList[T, L]{
		typeClient,
		alsoLister[T, L]{typeClient, emptyListCreator},
	}
}

// NewClientWithApply constructs a namespaced client with support for apply declarative configurations.
func NewClientWithApply[T objectWithMeta, C namedObject](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
	options ...Option[T],
) *ClientWithApply[T, C] {
	typeClient := NewClient[T](resource, client, parameterCodec, namespace, emptyObjectCreator, options...)
	return &ClientWithApply[T, C]{
		typeClient,
		alsoApplier[T, C]{typeClient},
	}
}

// NewClientWithListAndApply constructs a client with support for lists and applying declarative configurations.
func NewClientWithListAndApply[T objectWithMeta, L runtime.Object, C namedObject](
	resource string, client rest.Interface, parameterCodec runtime.ParameterCodec, namespace string, emptyObjectCreator func() T,
	emptyListCreator func() L, options ...Option[T],
) *ClientWithListAndApply[T, L, C] {
	typeClient := NewClient[T](resource, client, parameterCodec, namespace, emptyObjectCreator, options...)
	return &ClientWithListAndApply[T, L, C]{
		typeClient,
		alsoLister[T, L]{typeClient, emptyListCreator},
		alsoApplier[T, C]{typeClient},
	}
}

// GetClient returns the REST interface.
func (c *Client[T]) GetClient() rest.Interface {
	return c.client
}

// GetNamespace returns the client's namespace, if any.
func (c *Client[T]) GetNamespace() string {
	return c.namespace
}

// Get takes name of the resource, and returns the corresponding object, and an error if there is any.
func (c *Client[T]) Get(ctx context.Context, name string, options metav1.GetOptions) (T, error) {
	result, err := Get(ctx, &c.ResourceClient, c.newObject(), name, options)
	return any(result).(T), err
}

// Get takes name of the resource, and returns the corresponding object, and an error if there is any.
// T is the type managed by the client, R is the type returned by the server (usually the same).
func Get[R resource](ctx context.Context, c *ResourceClient, empty R, name string, options metav1.GetOptions) (*R, error) {
	return GetSubresource[R](ctx, c, name, "", options)
}

// Get takes name of the resource and subresource, and returns the corresponding object, and an error if there is any.
// T is the type managed by the client, R is the type of the subresource.
func GetSubresource[R resource](ctx context.Context, c *ResourceClient, name, subresource string, options metav1.GetOptions) (*R, error) {
	result := new(R)
	err := c.client.Get().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResourceIfNotEmpty(subresource).
		VersionedParams(&options, c.parameterCodec).
		Do(ctx).
		Into(resourceAsObject(result))
	return result, err
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func (l *alsoLister[T, L]) List(ctx context.Context, opts metav1.ListOptions) (L, error) {
	return List(ctx, l, opts)
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func List[T objectWithMeta, L runtime.Object](ctx context.Context, l *alsoLister[T, L], opts metav1.ListOptions) (L, error) {
	if watchListOptions, hasWatchListOptionsPrepared, watchListOptionsErr := watchlist.PrepareWatchListOptionsFromListOptions(opts); watchListOptionsErr != nil {
		klog.Warningf("Failed preparing watchlist options for $.type|resource$, falling back to the standard LIST semantics, err = %v", watchListOptionsErr)
	} else if hasWatchListOptionsPrepared {
		result, err := l.watchList(ctx, watchListOptions)
		if err == nil {
			consistencydetector.CheckWatchListFromCacheDataConsistencyIfRequested(ctx, "watchlist request for "+l.client.resource, l.list, opts, result)
			return result, nil
		}
		klog.Warningf("The watchlist request for %s ended with an error, falling back to the standard LIST semantics, err = %v", l.client.resource, err)
	}
	result, err := l.list(ctx, opts)
	if err == nil {
		consistencydetector.CheckListFromCacheDataConsistencyIfRequested(ctx, "list request for "+l.client.resource, l.list, opts, result)
	}
	return result, err
}

func (l *alsoLister[T, L]) list(ctx context.Context, opts metav1.ListOptions) (L, error) {
	return list(ctx, l, opts)
}

func list[T objectWithMeta, L runtime.Object](ctx context.Context, l *alsoLister[T, L], opts metav1.ListOptions) (L, error) {
	list := l.newList()
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	err := l.client.client.Get().
		UseProtobufAsDefaultIfPreferred(l.client.prefersProtobuf).
		NamespaceIfScoped(l.client.namespace, l.client.namespace != "").
		Resource(l.client.resource).
		VersionedParams(&opts, l.client.parameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(list)
	return list, err
}

// watchList establishes a watch stream with the server and returns the list of resources.
func (l *alsoLister[T, L]) watchList(ctx context.Context, opts metav1.ListOptions) (result L, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = l.newList()
	err = l.client.client.Get().
		UseProtobufAsDefaultIfPreferred(l.client.prefersProtobuf).
		NamespaceIfScoped(l.client.namespace, l.client.namespace != "").
		Resource(l.client.resource).
		VersionedParams(&opts, l.client.parameterCodec).
		Timeout(timeout).
		WatchList(ctx).
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested resources.
func (c *Client[T]) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.Get().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		VersionedParams(&opts, c.parameterCodec).
		Timeout(timeout).
		Watch(ctx)
}

// Create takes the representation of a resource and creates it.  Returns the server's representation of the resource, and an error, if there is any.
func (c *Client[T]) Create(ctx context.Context, obj T, opts metav1.CreateOptions) (T, error) {
	return Create(ctx, c, obj, c.newObject(), opts)
}

// Create takes the representation of a resource and creates it.  Returns the server's representation of the resource, and an error, if there is any.
// T is the type managed by the client, R is the type returned by the server (usually the same).
func Create[T objectWithMeta, I runtime.Object, R runtime.Object](ctx context.Context, c *Client[T], obj I, empty R, opts metav1.CreateOptions) (R, error) {
	err := c.client.Post().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(empty)
	return empty, err
}

// CreateSubresource takes the representation of a subresource and creates it.  Returns the server's representation of the resource, and an error, if there is any.
// T is the type managed by the client, I is the input type of the subresource, R is the subresource type returned by the server.
func CreateSubresource[T objectWithMeta, I runtime.Object, R runtime.Object](
	ctx context.Context, c *Client[T], name string, obj I, subresource string, empty R, opts metav1.CreateOptions,
) (R, error) {
	err := c.client.Post().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresource).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(empty)
	return empty, err
}

// Update takes the representation of a resource and updates it. Returns the server's representation of the resource, and an error, if there is any.
func (c *Client[T]) Update(ctx context.Context, obj T, opts metav1.UpdateOptions) (T, error) {
	return Update(ctx, c, obj, c.newObject(), opts)
}

// Update takes the representation of a resource and updates it. Returns the server's representation of the resource, and an error, if there is any.
// T is the type managed by the client, I is the input type, R is the type returned by the server.
func Update[T objectWithMeta, I objectWithMeta, R runtime.Object](ctx context.Context, c *Client[T], obj I, empty R, opts metav1.UpdateOptions) (R, error) {
	return UpdateSubresource(ctx, c, obj.GetName(), obj, "", empty, opts)
}

// Update takes the representation of a subresource and updates it. Returns the server's representation of the subresource, and an error, if there is any.
// T is the type managed by the client, I is the input type of the subresource, R is the subresource type returned by the server.
func UpdateSubresource[T objectWithMeta, I runtime.Object, R runtime.Object](
	ctx context.Context, c *Client[T], name string, obj I, subresource string, empty R, opts metav1.UpdateOptions,
) (R, error) {
	err := c.client.Put().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResourceIfNotEmpty(subresource).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(empty)
	return empty, err
}

// UpdateStatus updates the status subresource of a resource. Returns the server's representation of the resource, and an error, if there is any.
func (c *Client[T]) UpdateStatus(ctx context.Context, obj T, opts metav1.UpdateOptions) (T, error) {
	return UpdateSubresource(ctx, c, obj.GetName(), obj, "status", c.newObject(), opts)
}

// Delete takes name of the resource and deletes it. Returns an error if one occurs.
func (c *Client[T]) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	return c.client.Delete().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}

// DeleteCollection deletes a collection of objects.
func (l *alsoLister[T, L]) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	var timeout time.Duration
	if listOpts.TimeoutSeconds != nil {
		timeout = time.Duration(*listOpts.TimeoutSeconds) * time.Second
	}
	return l.client.client.Delete().
		UseProtobufAsDefaultIfPreferred(l.client.prefersProtobuf).
		NamespaceIfScoped(l.client.namespace, l.client.namespace != "").
		Resource(l.client.resource).
		VersionedParams(&listOpts, l.client.parameterCodec).
		Timeout(timeout).
		Body(&opts).
		Do(ctx).
		Error()
}

// Patch applies the patch and returns the patched resource.
func (c *Client[T]) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (T, error) {
	return Patch(ctx, c, name, pt, data, c.newObject(), opts, subresources...)
}

// Patch applies the patch and returns the patched resource.
func Patch[T objectWithMeta, R runtime.Object](ctx context.Context, c *Client[T], name string, pt types.PatchType, data []byte, empty R, opts metav1.PatchOptions, subresources ...string) (R, error) {
	err := c.client.Patch(pt).
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, c.parameterCodec).
		Body(data).
		Do(ctx).
		Into(empty)
	return empty, err
}

// Apply takes the given apply declarative configuration, applies it and returns the applied resource.
func (a *alsoApplier[T, C]) Apply(ctx context.Context, obj C, opts metav1.ApplyOptions) (T, error) {
	return Apply(ctx, a.client, obj, a.client.newObject(), opts)
}

// Apply takes the given apply declarative configuration, applies it and returns the applied resource.
// T is the type managed by the client, C is the apply configuration type, R is the type returned by the server.
func Apply[T objectWithMeta, C namedObject, R runtime.Object](ctx context.Context, c *Client[T], obj C, empty R, opts metav1.ApplyOptions) (R, error) {
	if obj.GetName() == nil {
		return *new(R), fmt.Errorf("obj.Name must be provided to Apply")
	}
	return ApplySubresource(ctx, c, *obj.GetName(), obj, "", empty, opts)
}

// Apply takes the given apply declarative configuration, applies it and returns the applied subresource.
// T is the type managed by the client, C is the apply configuration type, R is the subresource type returned by the server.
func ApplySubresource[T objectWithMeta, C comparable, R runtime.Object](
	ctx context.Context, c *Client[T], name string, obj C, subresource string, empty R, opts metav1.ApplyOptions,
) (R, error) {
	if obj == *new(C) {
		return *new(R), fmt.Errorf("object provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()

	request, err := apply.NewRequest(c.client, obj)
	if err != nil {
		return *new(R), err
	}

	err = request.
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResourceIfNotEmpty(subresource).
		VersionedParams(&patchOpts, c.parameterCodec).
		Do(ctx).
		Into(empty)
	return empty, err
}

// Apply takes the given apply declarative configuration, applies it to the status subresource and returns the applied resource.
func (a *alsoApplier[T, C]) ApplyStatus(ctx context.Context, obj C, opts metav1.ApplyOptions) (T, error) {
	if obj.GetName() == nil {
		return *new(T), fmt.Errorf("obj.Name must be provided to ApplyStatus")
	}
	return ApplySubresource(ctx, a.client, *obj.GetName(), obj, "status", a.client.newObject(), opts)
}
