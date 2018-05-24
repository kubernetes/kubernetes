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

package openapi

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
)

// SpecSource is an interface for fetching the openapi spec and parsing it into an Resources struct
type SpecSource interface {
	// Get returns the entire parsed OpenAPI spec
	Get() (Resources, error)
}

// Resources interface describe a resources provider, that can give you
// resource based on group-version-kind.
type Resources interface {
	// LookupResource looks up the OpenAPI schema for a particular gvk
	LookupResource(gvk schema.GroupVersionKind) (proto.Schema, error)
}

// SpecDownloader sepcifies where to download an OpenAPI schema from
type SpecDownloader interface {
	// Download attempts to get the OpenAPI spec if it has changed.
	Download(lastEtag string) (
		specBytes []byte,
		newEtag string,
		httpStatus int,
		err error)
}

// SpecParser is to here to make unit testing easier
type SpecParser interface {
	// Parse converts a byte array to a Resouces object for easier schema lookups
	Parse(raw []byte) (Resources, error)
}

// ResourceSource is an interface for fetching the openapi schema for a single resource
type ResourceSource interface {
	Get() (proto.Schema, error)
}
