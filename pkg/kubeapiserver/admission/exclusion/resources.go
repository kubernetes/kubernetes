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

package exclusion

import (
	"slices"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// include is the list of resources that the expression-based admission controllers
// should intercept.
// The version is omitted, all versions of the same GroupResource are treated the same.
// If a resource is transient, i.e., not persisted in the storage, the resource must be
// in either include or excluded list.
var included = []schema.GroupResource{
	{Group: "", Resource: "bindings"},
	{Group: "", Resource: "pods/attach"},
	{Group: "", Resource: "pods/binding"},
	{Group: "", Resource: "pods/eviction"},
	{Group: "", Resource: "pods/exec"},
	{Group: "", Resource: "pods/portforward"},

	// ref: https://github.com/kubernetes/kubernetes/issues/122205#issuecomment-1927390823
	{Group: "", Resource: "serviceaccounts/token"},
}

// excluded is the list of resources that the expression-based admission controllers
// should ignore.
// The version is omitted, all versions of the same GroupResource are treated the same.
var excluded = []schema.GroupResource{
	// BEGIN interception of these non-persisted resources may break the cluster
	{Group: "authentication.k8s.io", Resource: "selfsubjectreviews"},
	{Group: "authentication.k8s.io", Resource: "tokenreviews"},
	{Group: "authorization.k8s.io", Resource: "localsubjectaccessreviews"},
	{Group: "authorization.k8s.io", Resource: "selfsubjectaccessreviews"},
	{Group: "authorization.k8s.io", Resource: "selfsubjectrulesreviews"},
	{Group: "authorization.k8s.io", Resource: "subjectaccessreviews"},
	// END interception of these non-persisted resources may break the cluster
}

// Included returns a copy of the list of resources that the expression-based admission controllers
// should intercept.
func Included() []schema.GroupResource {
	return slices.Clone(included)
}

// Excluded returns a copy of the list of resources that the expression-based admission controllers
// should ignore.
func Excluded() []schema.GroupResource {
	return slices.Clone(excluded)
}
