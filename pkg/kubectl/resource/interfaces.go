/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// RESTClient is a client helper for dealing with RESTful resources
// in a generic way.
type RESTClient interface {
	Get() *client.Request
	Post() *client.Request
	Patch(api.PatchType) *client.Request
	Delete() *client.Request
	Put() *client.Request
}

// ClientMapper retrieves a client object for a given mapping
type ClientMapper interface {
	ClientForMapping(mapping *meta.RESTMapping) (RESTClient, error)
}

// ClientMapperFunc implements ClientMapper for a function
type ClientMapperFunc func(mapping *meta.RESTMapping) (RESTClient, error)

// ClientForMapping implements ClientMapper
func (f ClientMapperFunc) ClientForMapping(mapping *meta.RESTMapping) (RESTClient, error) {
	return f(mapping)
}

// StreamTransform is a function that takes a reader, and returns a reader that transforms the input
// in some way.  For example, transforming a simplified YAML file into a complete API object.
type StreamTransform func(io.Reader) (io.Reader, error)

// StreamTransformer is an interface for things that know how to accept StreamTransforms to transform
// their input.
type StreamTransformer interface {
	SetStreamTransform(fn StreamTransform)
}
