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

package v2

import (
	"context"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/apply"
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

// TypedClient represents a client, optionally namespaced, with no support for lists or apply declarative configurations.
type TypedClient[T any, PT objectWithMeta[T]] struct {
	Client
}

type Client struct {
	resource        string
	client          rest.Interface
	namespace       string // "" for non-namespaced clients
	parameterCodec  runtime.ParameterCodec
	prefersProtobuf bool
}

// TypedClientWithList represents a client with support for lists.
type TypedClientWithList[T, L any, PT objectWithMeta[T], PL runtimeObject[L]] struct {
	*TypedClient[T, PT]
	alsoLister[T, L, PT, PL]
}

// TypedClientWithApply represents a client with support for apply declarative configurations.
type TypedClientWithApply[T, C any, PT objectWithMeta[T], PC namedPointerObject[C]] struct {
	*TypedClient[T, PT]
	alsoApplier[T, C, PT, PC]
}

// TypedClientWithListAndApply represents a client with support for lists and apply declarative configurations.
type TypedClientWithListAndApply[T, L, C any, PT objectWithMeta[T], PL runtimeObject[L], PC namedPointerObject[C]] struct {
	*TypedClient[T, PT]
	alsoLister[T, L, PT, PL]
	alsoApplier[T, C, PT, PC]
}

// Helper types for composition
type alsoLister[T, L any, PT objectWithMeta[T], PL runtimeObject[L]] struct {
	*TypedClient[T, PT]
}

type alsoApplier[T, C any, PT objectWithMeta[T], PC namedPointerObject[C]] struct {
	*TypedClient[T, PT]
}

type Option func(*Client)

// ForResource declares the resource on which the client operates.
func ForResource(resource string) Option {
	return func(c *Client) { c.resource = resource }
}

// InNamespace specifies the namespace in which the client operates.
func InNamespace(namespace string) Option {
	return func(c *Client) { c.namespace = namespace }
}

// UsingClient provides a REST interface for the client to use.
func UsingClient(client rest.Interface) Option {
	return func(c *Client) { c.client = client }
}

// UsingParameterCodec provides a parameter codec for the client to use.
func UsingParameterCodec(parameterCodec runtime.ParameterCodec) Option {
	return func(c *Client) { c.parameterCodec = parameterCodec }
}

// PrefersProtobuf specifies that the client should prefer protobuf.
func PrefersProtobuf() Option {
	return func(c *Client) { c.prefersProtobuf = true }
}

// NewTypedClient constructs a client, namespaced or not, with no support for lists or apply.
func NewTypedClient[T any, PT objectWithMeta[T]](opts ...Option) *TypedClient[T, PT] {
	c := &TypedClient[T, PT]{}
	for _, option := range opts {
		option(&c.Client)
	}
	return c
}

// NewTypedClientWithList constructs a client with support for lists.
func NewTypedClientWithList[T, L any, PT objectWithMeta[T], PL runtimeObject[L]](opts ...Option) *TypedClientWithList[T, L, PT, PL] {
	typeClient := NewTypedClient[T, PT](opts...)
	return &TypedClientWithList[T, L, PT, PL]{
		typeClient,
		alsoLister[T, L, PT, PL]{typeClient},
	}
}

// NewTypedClientWithApply constructs a client with support for apply declarative configurations.
func NewTypedClientWithApply[T, C any, PT objectWithMeta[T], PC namedPointerObject[C]](opts ...Option) *TypedClientWithApply[T, C, PT, PC] {
	typeClient := NewTypedClient[T, PT](opts...)
	return &TypedClientWithApply[T, C, PT, PC]{
		typeClient,
		alsoApplier[T, C, PT, PC]{typeClient},
	}
}

// NewTypedClientWithListAndApply constructs a client with support for lists and applying declarative configurations.
func NewTypedClientWithListAndApply[T, L, C any, PT objectWithMeta[T], PL runtimeObject[L], PC namedPointerObject[C]](
	opts ...Option,
) *TypedClientWithListAndApply[T, L, C, PT, PL, PC] {
	typeClient := NewTypedClient[T, PT](opts...)
	return &TypedClientWithListAndApply[T, L, C, PT, PL, PC]{
		typeClient,
		alsoLister[T, L, PT, PL]{typeClient},
		alsoApplier[T, C, PT, PC]{typeClient},
	}
}

// GetClient returns the REST interface.
func (c *TypedClient[T, PT]) GetClient() rest.Interface {
	return c.client
}

// GetNamespace returns the client's namespace, if any.
func (c *TypedClient[T, PT]) GetNamespace() string {
	return c.namespace
}

// Get takes name of the resource, and returns the corresponding object, and an error if there is any.
func (c *TypedClient[T, PT]) Get(ctx context.Context, name string, opts metav1.GetOptions) (PT, error) {
	return c.Client.Get[T, PT](ctx, name, opts)
}

// Get takes name of the resource, and returns the corresponding object, and an error if there is any.
func (c *Client) Get[T any, PT runtimeObject[T]](ctx context.Context, name string, opts metav1.GetOptions) (PT, error) {
	result := PT(new(T))
	err := c.client.Get().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		VersionedParams(&opts, c.parameterCodec).
		Do(ctx).
		Into(result)
	return result, err
}

// SubResourceGet takes name of the resource and subresource, and returns the corresponding subresource, and an error if there is any.
func (c *Client) SubResourceGet[S any, PS runtimeObject[S]](ctx context.Context, name, subresource string, opts metav1.GetOptions) (PS, error) {
	result := PS(new(S))
	err := c.client.Get().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresource).
		VersionedParams(&opts, c.parameterCodec).
		Do(ctx).
		Into(result)
	return result, err
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func (l *alsoLister[T, L, PT, PL]) List(ctx context.Context, opts metav1.ListOptions) (PL, error) {
	return l.Client.List[L, PL](ctx, opts)
}

// List takes label and field selectors, and returns the list of resources that match those selectors.
func (c *Client) List[L any, PL runtimeObject[L]](ctx context.Context, opts metav1.ListOptions) (PL, error) {
	list := PL(new(L))
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	err := c.client.Get().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		VersionedParams(&opts, c.parameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(list)
	return list, err
}

// SubResourceList takes label and field selectors, resource and subresource names, and returns the list of subresources that match those selectors.
func (c *Client) SubResourceList[L any, PL runtimeObject[L]](ctx context.Context, name, subresource string, opts metav1.ListOptions) (PL, error) {
	list := PL(new(L))
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	err := c.client.Get().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresource).
		VersionedParams(&opts, c.parameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(list)
	return list, err
}

// Watch returns a watch.Interface that watches the requested resources.
func (c *Client) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
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
func (c *TypedClient[T, PT]) Create(ctx context.Context, obj PT, opts metav1.CreateOptions) (PT, error) {
	return c.Client.Create(ctx, obj, opts)
}

// Create takes the representation of a resource and creates it.  Returns the server's representation of the resource, and an error, if there is any.
func (c *Client) Create[T any, PT runtimeObject[T]](ctx context.Context, obj PT, opts metav1.CreateOptions) (PT, error) {
	result := PT(new(T))
	err := c.client.Post().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(result)
	return result, err
}

// SubResourceCreate takes the representation of a subresource and creates it.  Returns the server's representation of the subresource, and an error, if there is any.
func (c *Client) SubResourceCreate[S any, PS runtimeObject[S]](ctx context.Context, name, subresource string, obj PS, opts metav1.CreateOptions) (PS, error) {
	result := PS(new(S))
	err := c.client.Post().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresource).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(result)
	return result, err
}

// Update takes the representation of a resource and updates it. Returns the server's representation of the resource, and an error, if there is any.
func (c *TypedClient[T, PT]) Update(ctx context.Context, obj PT, opts metav1.UpdateOptions) (PT, error) {
	return c.Client.Update(ctx, obj, opts)
}

// Update takes the representation of a resource and updates it. Returns the server's representation of the resource, and an error, if there is any.
func (c *Client) Update[T any, PT objectWithMeta[T]](ctx context.Context, obj PT, opts metav1.UpdateOptions) (PT, error) {
	result := PT(new(T))
	err := c.client.Put().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(obj.GetName()).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(result)
	return result, err
}

// SubResourceUpdate takes the representation of a subresource and updates it.  Returns the server's representation of the subresource, and an error, if there is any.
func (c *Client) SubResourceUpdate[S any, PS runtimeObject[S]](ctx context.Context, name, subresource string, obj PS, opts metav1.UpdateOptions) (PS, error) {
	result := PS(new(S))
	err := c.client.Put().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresource).
		VersionedParams(&opts, c.parameterCodec).
		Body(obj).
		Do(ctx).
		Into(result)
	return result, err
}

// UpdateStatus updates the status subresource of a resource. Returns the server's representation of the resource, and an error, if there is any.
func (c *TypedClient[T, PT]) UpdateStatus(ctx context.Context, obj PT, opts metav1.UpdateOptions) (PT, error) {
	return c.SubResourceUpdate(ctx, obj.GetName(), "status", obj, opts)
}

// Delete takes name of the resource and deletes it. Returns an error if one occurs.
func (c *Client) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	return c.client.Delete().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}

// DeleteWithResult takes name of the resource and deletes it. Returns an error if one occurs, or a metav1.APIResult on success.
func (c *Client) DeleteWithResult(ctx context.Context, name string, opts metav1.DeleteOptions) (metav1.APIResult, error) {
	result := c.client.Delete().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		Body(&opts).
		Do(ctx)
	return rest.APIResultAdapter{Result: result}, result.Error()
}

// DeleteCollection deletes a collection of objects.
func (c *Client) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	var timeout time.Duration
	if listOpts.TimeoutSeconds != nil {
		timeout = time.Duration(*listOpts.TimeoutSeconds) * time.Second
	}
	return c.client.Delete().
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		VersionedParams(&listOpts, c.parameterCodec).
		Timeout(timeout).
		Body(&opts).
		Do(ctx).
		Error()
}

// Patch applies the patch and returns the patched resource.
func (c *TypedClient[T, PT]) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (PT, error) {
	return c.Client.Patch[T, PT](ctx, name, pt, data, opts, subresources...)
}

// Patch applies the patch and returns the patched resource.
func (c *Client) Patch[T any, PT runtimeObject[T]](ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (PT, error) {
	result := PT(new(T))
	err := c.client.Patch(pt).
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, c.parameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return result, err
}

// Apply takes the given apply declarative configuration, applies it and returns the applied resource.
func (a *alsoApplier[T, C, PT, PC]) Apply(ctx context.Context, obj PC, opts metav1.ApplyOptions) (PT, error) {
	return a.Client.Apply[T, C, PT](ctx, obj, opts)
}

// Apply takes the given apply declarative configuration, applies it and returns the applied resource.
func (c *Client) Apply[T, C any, PT runtimeObject[T], PC namedPointerObject[C]](ctx context.Context, obj PC, opts metav1.ApplyOptions) (PT, error) {
	result := PT(new(T))
	if obj == nil {
		return result, fmt.Errorf("object provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	if obj.GetName() == nil {
		return result, fmt.Errorf("obj.Name must be provided to Apply")
	}

	request, err := apply.NewRequest(c.client, obj)
	if err != nil {
		return result, err
	}

	err = request.
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(*obj.GetName()).
		VersionedParams(&patchOpts, c.parameterCodec).
		Do(ctx).
		Into(result)
	return result, err
}

// Apply takes the given apply declarative configuration, applies it and returns the applied subresource.
func (c *Client) SubResourceApply[S, C any, PS runtimeObject[S], PC *C](ctx context.Context, name, subresource string, obj PC, opts metav1.ApplyOptions) (PS, error) {
	result := PS(new(S))
	if obj == nil {
		return result, fmt.Errorf("object provided to Apply must not be nil")
	}
	patchOpts := opts.ToPatchOptions()
	request, err := apply.NewRequest(c.client, obj)
	if err != nil {
		return result, err
	}

	err = request.
		UseProtobufAsDefaultIfPreferred(c.prefersProtobuf).
		NamespaceIfScoped(c.namespace, c.namespace != "").
		Resource(c.resource).
		Name(name).
		SubResource(subresource).
		VersionedParams(&patchOpts, c.parameterCodec).
		Do(ctx).
		Into(result)
	return result, err
}

// ApplyStatus takes the given apply declarative configuration, applies it to the status subresource and returns the applied resource.
func (a *alsoApplier[T, C, PT, PC]) ApplyStatus(ctx context.Context, obj PC, opts metav1.ApplyOptions) (PT, error) {
	return a.SubResourceApply[T, C, PT](ctx, *obj.GetName(), "status", obj, opts)
}
