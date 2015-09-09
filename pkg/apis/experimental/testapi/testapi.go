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

package testapi

import (
	"strings"

	"k8s.io/kubernetes/pkg/apis/experimental/latest"
)

// Returns the appropriate path for the given prefix (watch, proxy, redirect, etc), resource, namespace and name.
// For example, this is of the form:
// /experimental/v1/watch/namespaces/foo/pods/pod0 for v1.
func ResourcePathWithPrefix(prefix, resource, namespace, name string) string {
	path := "/experimental/" + latest.Version
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
// /experimental/v1/namespaces/foo/pods/pod0 for v1.
func ResourcePath(resource, namespace, name string) string {
	return ResourcePathWithPrefix("", resource, namespace, name)
}
