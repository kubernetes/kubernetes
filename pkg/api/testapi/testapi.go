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

// Package testapi provides a helper for retrieving the KUBE_API_VERSION environment variable.
package testapi

import (
	"fmt"
	"os"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Version returns the API version to test against, as set by the KUBE_API_VERSION env var.
func Version() string {
	version := os.Getenv("KUBE_API_VERSION")
	if version == "" {
		version = latest.Version
	}
	return version
}

// Codec returns the codec for the API version to test against, as set by the
// KUBE_API_VERSION env var.
func Codec() runtime.Codec {
	interfaces, err := latest.InterfacesFor(Version())
	if err != nil {
		panic(err)
	}
	return interfaces.Codec
}

// Converter returns the api.Scheme for the API version to test against, as set by the
// KUBE_API_VERSION env var.
func Converter() runtime.ObjectConvertor {
	interfaces, err := latest.InterfacesFor(Version())
	if err != nil {
		panic(err)
	}
	return interfaces.ObjectConvertor
}

// MetadataAccessor returns the MetadataAccessor for the API version to test against,
// as set by the KUBE_API_VERSION env var.
func MetadataAccessor() meta.MetadataAccessor {
	interfaces, err := latest.InterfacesFor(Version())
	if err != nil {
		panic(err)
	}
	return interfaces.MetadataAccessor
}

// SelfLink returns a self link that will appear to be for the version Version().
// 'resource' should be the resource path, e.g. "pods" for the Pod type. 'name' should be
// empty for lists.
func SelfLink(resource, name string) string {
	if name == "" {
		return fmt.Sprintf("/api/%s/%s", Version(), resource)
	}
	return fmt.Sprintf("/api/%s/%s/%s", Version(), resource, name)
}

// Returns the appropriate path for the given prefix (watch, proxy, redirect, etc), resource, namespace and name.
// For ex, this is of the form:
// /api/v1/watch/namespaces/foo/pods/pod0 for v1.
func ResourcePathWithPrefix(prefix, resource, namespace, name string) string {
	path := "/api/" + Version()
	if prefix != "" {
		path = path + "/" + prefix
	}
	if namespace != "" {
		path = path + "/namespaces/" + namespace
	}
	// Resource names are lower case.
	resource = strings.ToLower(resource)
	if resource != "" {
		path = path + "/" + resource
	}
	if name != "" {
		path = path + "/" + name
	}
	return path
}

// Returns the appropriate path for the given resource, namespace and name.
// For ex, this is of the form:
// /api/v1beta1/pods/pod0 for v1beta1 and
// /api/v1beta3/namespaces/foo/pods/pod0 for v1beta3.
func ResourcePath(resource, namespace, name string) string {
	return ResourcePathWithPrefix("", resource, namespace, name)
}
