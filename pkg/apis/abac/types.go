/*
Copyright 2015 The Kubernetes Authors.

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

package abac

import "k8s.io/kubernetes/pkg/api/unversioned"

// Policy contains a single ABAC policy rule
type Policy struct {
	unversioned.TypeMeta

	// Spec describes the policy rule
	Spec PolicySpec
}

// PolicySpec contains the attributes for a policy rule
type PolicySpec struct {

	// User is the username this rule applies to.
	// Either user or group is required to match the request.
	// "*" matches all users.
	User string

	// Group is the group this rule applies to.
	// Either user or group is required to match the request.
	// "*" matches all groups.
	Group string

	// Readonly matches readonly requests when true, and all requests when false
	Readonly bool

	// APIGroup is the name of an API group. APIGroup, Resource, and Namespace are required to match resource requests.
	// "*" matches all API groups
	APIGroup string

	// Resource is the name of a resource. APIGroup, Resource, and Namespace are required to match resource requests.
	// "*" matches all resources
	Resource string

	// Namespace is the name of a namespace. APIGroup, Resource, and Namespace are required to match resource requests.
	// "*" matches all namespaces (including unnamespaced requests)
	Namespace string

	// NonResourcePath matches non-resource request paths.
	// "*" matches all paths
	// "/foo/*" matches all subpaths of foo
	NonResourcePath string

	// TODO: "expires" string in RFC3339 format.

	// TODO: want a way to allow some users to restart containers of a pod but
	// not delete or modify it.

	// TODO: want a way to allow a controller to create a pod based only on a
	// certain podTemplates.

}
