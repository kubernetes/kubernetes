/*
Copyright 2017 The Kubernetes Authors.

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

package resourcequota

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// Configuration provides configuration for the ResourceQuota admission controller.
type Configuration struct {
	metav1.TypeMeta

	// LimitedResources whose consumption is limited by default.
	// +optional
	LimitedResources []LimitedResource
}

// LimitedResource matches a resource whose consumption is limited by default.
// To consume the resource, there must exist an associated quota that limits
// its consumption.
type LimitedResource struct {
	// API group of the object whose consumption should be limited.
	// The quota system will intercept creation of all objects that
	// match the specified APIGroup.
	// +optional
	APIGroup string

	// API resource of the object to match.
	// The quota system will intercept creation of all objects that
	// match the specified APIResource.
	// For example, if the administrator wants to limit consumption
	// of a compute resource associated with pod objects, the value
	// would be "pods".
	APIResource string

	// If a resource name tracked by quota for this entity matches this expression, and there is
	// no associated quota for that resource name defined, the request will be denied by the quota
	// system.  The MatchExpression is a regular expression value according to
	// https://golang.org/pkg/regexp/
	// For example, if an administrator wants to globally enforce that
	// a quota must exist to consume cpu and memory associated with all pods
	// the match expression value would be "^(requests.cpu|requests.memory)$"
	MatchExpression string
}
