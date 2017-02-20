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

	// APIGroup is the name of the APIGroup that contains the limited resource.
	// +optional
	APIGroup string `json:"apiGroup,omitempty"`

	// Resource is the name of the resource this rule applies to.
	// For example, if the administrator wants to limit consumption
	// of a storage resource associated with persistent volume claims,
	// the value would be "persistentvolumeclaims".
	Resource string `json:"resource"`

	// For each intercepted request, the quota system will evaluate
	// its resource usage.  It will iterate through each resource consumed
	// and if the resource contains any substring in this listing, the
	// quota system will ensure that there is a covering quota.  In the
	// absence of a covering quota, the quota system will deny the request.
	// For example, if an administrator wants to globally enforce that
	// that a quota must exist to consume persistent volume claims associated
	// with any storage class, the list would include
	// ".storageclass.storage.k8s.io/requests.storage"
	MatchContains []string
}
