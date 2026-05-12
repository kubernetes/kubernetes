/*
Copyright 2014 The Kubernetes Authors.

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

package resource

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
)

type RESTClientGetter interface {
	ToRESTConfig() (*rest.Config, error)
	ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error)
	ToRESTMapper() (meta.RESTMapper, error)
}

type ClientConfigFunc func() (*rest.Config, error)
type RESTMapperFunc func() (meta.RESTMapper, error)
type CategoryExpanderFunc func() (restmapper.CategoryExpander, error)

// RESTClient is a client helper for dealing with RESTful resources
// in a generic way.
type RESTClient interface {
	Get() *rest.Request
	Post() *rest.Request
	Patch(types.PatchType) *rest.Request
	Delete() *rest.Request
	Put() *rest.Request
}

// RequestTransform is a function that is given a chance to modify the outgoing request.
type RequestTransform func(*rest.Request)

// NewClientWithOptions wraps the provided RESTClient and invokes each transform on each
// newly created request.
func NewClientWithOptions(c RESTClient, transforms ...RequestTransform) RESTClient {
	if len(transforms) == 0 {
		return c
	}
	return &clientOptions{c: c, transforms: transforms}
}

type clientOptions struct {
	c          RESTClient
	transforms []RequestTransform
}

func (c *clientOptions) modify(req *rest.Request) *rest.Request {
	for _, transform := range c.transforms {
		transform(req)
	}
	return req
}

func (c *clientOptions) Get() *rest.Request {
	return c.modify(c.c.Get())
}

func (c *clientOptions) Post() *rest.Request {
	return c.modify(c.c.Post())
}
func (c *clientOptions) Patch(t types.PatchType) *rest.Request {
	return c.modify(c.c.Patch(t))
}
func (c *clientOptions) Delete() *rest.Request {
	return c.modify(c.c.Delete())
}
func (c *clientOptions) Put() *rest.Request {
	return c.modify(c.c.Put())
}

// ContentValidator is an interface that knows how to validate an API object serialized to a byte array.
type ContentValidator interface {
	ValidateBytes(data []byte) error
}

// Visitor lets clients walk a list of resources.
type Visitor interface {
	Visit(VisitorFunc) error
}

// VisitorFunc implements the Visitor interface for a matching function.
// If there was a problem walking a list of resources, the incoming error
// will describe the problem and the function can decide how to handle that error.
// A nil returned indicates to accept an error to continue loops even when errors happen.
// This is useful for ignoring certain kinds of errors or aggregating errors in some way.
type VisitorFunc func(*Info, error) error
