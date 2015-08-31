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

// Package testapi provides a helper for retrieving the KUBE_TEST_API environment variable.
package testapi

import (
	"fmt"
	"os"
	"strings"

	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/meta"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	explatest "k8s.io/kubernetes/pkg/expapi/latest"
	"k8s.io/kubernetes/pkg/runtime"
)

// GroupAndVersion returns the API version to test against for a group, as set
// by the KUBE_TEST_API env var.
// Return value is in the form of "group/version".
// If there is no group input, return "v1".
func GroupAndVersion(group ...string) string {
	if len(group) == 0 {
		return "v1"
	}

	version := os.Getenv("KUBE_TEST_API")
	testGroupVersions := strings.Split(version, ",")

	for _, groupVersion := range testGroupVersions {
		if apiutil.GetGroup(groupVersion) == group[0] {
			return groupVersion
		}
	}
	//TODO: we need a central place to store all available API groups and their metadata in their respective latest.go
	if group[0] == "experimental" {
		return explatest.GroupVersion
	} else {
		panic(fmt.Errorf("cannot find appropriate testing version for group %s", group))
	}
}

// Version implements the same logic as GroupAndVersion but the return value is
// in the form of "version"
func Version(group ...string) string {
	return apiutil.GetVersion(GroupAndVersion(group...))
}

// Codec returns the codec for the API version of the group to test against, as
// set by the KUBE_TEST_API env var.
func Codec(group ...string) runtime.Codec {
	if len(group) == 0 || group[0] == "" {
		interfaces, err := latest.InterfacesFor(Version())
		if err != nil {
			panic(err)
		}
		return interfaces.Codec
	}
	if group[0] == "experimental" {
		interfaces, err := explatest.InterfacesFor(Version(group...))
		if err != nil {
			panic(err)
		}
		return interfaces.Codec
	}
	panic(fmt.Errorf("cannot test group %s", group))
}

// Converter returns the api.Scheme for the API version of the group to test
// against, as set by the KUBE_TEST_API env var.
func Converter(group ...string) runtime.ObjectConvertor {
	if len(group) == 0 || group[0] == "" {
		interfaces, err := latest.InterfacesFor(Version())
		if err != nil {
			panic(err)
		}
		return interfaces.ObjectConvertor
	}
	if group[0] == "experimental" {
		interfaces, err := explatest.InterfacesFor(Version(group...))
		if err != nil {
			panic(err)
		}
		return interfaces.ObjectConvertor
	}
	panic(fmt.Errorf("cannot test group %s", group))
}

// MetadataAccessor returns the MetadataAccessor for the API version of the
// group to test against, as set by the KUBE_TEST_API env var.
func MetadataAccessor(group ...string) meta.MetadataAccessor {
	if len(group) == 0 || group[0] == "" {
		interfaces, err := latest.InterfacesFor(Version())
		if err != nil {
			panic(err)
		}
		return interfaces.MetadataAccessor
	}
	if group[0] == "experimental" {
		interfaces, err := explatest.InterfacesFor(Version(group...))
		if err != nil {
			panic(err)
		}
		return interfaces.MetadataAccessor
	}
	panic(fmt.Errorf("cannot test group %s", group))
}

// SelfLink returns a self link that will appear to be for the version
// Version(group...).
// 'resource' should be the resource path, e.g. "pods" for the Pod type. 'name' should be
// empty for lists.
func SelfLink(resource, name string, group ...string) string {
	if name == "" {
		return fmt.Sprintf("/api/%s/%s", Version(group...), resource)
	}
	return fmt.Sprintf("/api/%s/%s/%s", Version(group...), resource, name)
}

// Returns the appropriate path for the given prefix (watch, proxy, redirect, etc), resource, namespace and name.
// For ex, this is of the form:
// /api/v1/watch/namespaces/foo/pods/pod0 for v1.
// TODO: needs to change the prefix for API groups other than v1.
func ResourcePathWithPrefix(prefix, resource, namespace, name string, group ...string) string {
	var path string
	if len(group) == 0 {
		path = "/api/" + Version()
	} else {
		path = "/api/" + group[0] + "/" + Version(group...)
	}

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
// For example, this is of the form:
// /api/v1/namespaces/foo/pods/pod0 for v1.
func ResourcePath(resource, namespace, name string, group ...string) string {
	return ResourcePathWithPrefix("", resource, namespace, name, group...)
}
