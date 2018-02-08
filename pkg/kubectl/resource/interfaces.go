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
	client "k8s.io/client-go/rest"
)

// RESTClient is a client helper for dealing with RESTful resources
// in a generic way.
type RESTClient interface {
	Get() *client.Request
	Post() *client.Request
	Patch(types.PatchType) *client.Request
	Delete() *client.Request
	Put() *client.Request
}

// ClientMapper abstracts retrieving a Client for mapped objects.
type ClientMapper interface {
	ClientForMapping(mapping *meta.RESTMapping) (RESTClient, error)
}

// ClientMapperFunc implements ClientMapper for a function
type ClientMapperFunc func(mapping *meta.RESTMapping) (RESTClient, error)

// ClientForMapping implements ClientMapper
func (f ClientMapperFunc) ClientForMapping(mapping *meta.RESTMapping) (RESTClient, error) {
	return f(mapping)
}

// RequestTransform is a function that is given a chance to modify the outgoing request.
type RequestTransform func(*client.Request)

// NewClientWithOptions wraps the provided RESTClient and invokes each transform on each
// newly created request.
func NewClientWithOptions(c RESTClient, transforms ...RequestTransform) RESTClient {
	return &clientOptions{c: c, transforms: transforms}
}

type clientOptions struct {
	c          RESTClient
	transforms []RequestTransform
}

func (c *clientOptions) modify(req *client.Request) *client.Request {
	for _, transform := range c.transforms {
		transform(req)
	}
	return req
}

func (c *clientOptions) Get() *client.Request {
	return c.modify(c.c.Get())
}

func (c *clientOptions) Post() *client.Request {
	return c.modify(c.c.Post())
}
func (c *clientOptions) Patch(t types.PatchType) *client.Request {
	return c.modify(c.c.Patch(t))
}
func (c *clientOptions) Delete() *client.Request {
	return c.modify(c.c.Delete())
}
func (c *clientOptions) Put() *client.Request {
	return c.modify(c.c.Put())
}
