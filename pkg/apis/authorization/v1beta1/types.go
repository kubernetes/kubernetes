/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1beta1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// SubjectAccessReview checks whether or not a user or group can perform an action.
type SubjectAccessReview struct {
	unversioned.TypeMeta `json:",inline"`

	// Spec holds information about the request being evaluated
	Spec SubjectAccessReviewSpec `json:"spec"`

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus `json:"status,omitempty"`
}

// SelfSubjectAccessReview checks whether or the current user can perform an action.  Not filling in a
// spec.namespace means "in all namespaces".  Self is a special case, because users should always be able
// to check whether they can perform an action
type SelfSubjectAccessReview struct {
	unversioned.TypeMeta `json:",inline"`

	// Spec holds information about the request being evaluated.  user and groups must be empty
	Spec SelfSubjectAccessReviewSpec `json:"spec"`

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus `json:"status,omitempty"`
}

// LocalSubjectAccessReview checks whether or not a user or group can perform an action in a given namespace.
// Having a namespace scoped resource makes it much easier to grant namespace scoped policy that includes permissions
// checking.
type LocalSubjectAccessReview struct {
	unversioned.TypeMeta `json:",inline"`

	// Spec holds information about the request being evaluated.  spec.namespace must be equal to the namespace
	// you made the request against.  If empty, it is defaulted.
	Spec SubjectAccessReviewSpec `json:"spec"`

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus `json:"status,omitempty"`
}

// ResourceAttributes includes the authorization attributes available for resource requests to the Authorizer interface
type ResourceAttributes struct {
	// Namespace is the namespace of the action being requested.  Currently, there is no distinction between no namespace and all namespaces
	// "" (empty) is defaulted for LocalSubjectAccessReviews
	// "" (empty) is empty for cluster-scoped resources
	// "" (empty) means "all" for namespace scoped resources from a SubjectAccessReview or SelfSubjectAccessReview
	Namespace string `json:"namespace,omitempty"`
	// Verb is a kubernetes resource API verb, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	Verb string `json:"verb,omitempty"`
	// Group is the API Group of the Resource.  "*" means all.
	Group string `json:"group,omitempty"`
	// Version is the API Version of the Resource.  "*" means all.
	Version string `json:"version,omitempty"`
	// Resource is one of the existing resource types.  "*" means all.
	Resource string `json:"resource,omitempty"`
	// Subresource is one of the existing resource types.  "" means none.
	Subresource string `json:"subresource,omitempty"`
	// Name is the name of the resource being requested for a "get" or deleted for a "delete". "" (empty) means all.
	Name string `json:"name,omitempty"`
}

// NonResourceAttributes includes the authorization attributes available for non-resource requests to the Authorizer interface
type NonResourceAttributes struct {
	// Path is the URL path of the request
	Path string `json:"path,omitempty"`
	// Verb is the standard HTTP verb
	Verb string `json:"verb,omitempty"`
}

// SubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAuthorizationAttributes
// and NonResourceAuthorizationAttributes must be set
type SubjectAccessReviewSpec struct {
	// ResourceAuthorizationAttributes describes information for a resource access request
	ResourceAttributes *ResourceAttributes `json:"resourceAttributes,omitempty"`
	// NonResourceAttributes describes information for a non-resource access request
	NonResourceAttributes *NonResourceAttributes `json:"nonResourceAttributes,omitempty"`

	// User is the user you're testing for.
	// If you specify "User" but not "Group", then is it interpreted as "What if User were not a member of any groups
	User string `json:"user,omitempty"`
	// Groups is the groups you're testing for.
	Groups []string `json:"group,omitempty"`
	// Extra corresponds to the user.Info.GetExtra() method from the authenticator.  Since that is input to the authorizer
	// it needs a reflection here.
	Extra map[string][]string `json:"extra,omitempty"`
}

// SelfSubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAuthorizationAttributes
// and NonResourceAuthorizationAttributes must be set
type SelfSubjectAccessReviewSpec struct {
	// ResourceAuthorizationAttributes describes information for a resource access request
	ResourceAttributes *ResourceAttributes `json:"resourceAttributes,omitempty"`
	// NonResourceAttributes describes information for a non-resource access request
	NonResourceAttributes *NonResourceAttributes `json:"nonResourceAttributes,omitempty"`
}

// SubjectAccessReviewStatus
type SubjectAccessReviewStatus struct {
	// Allowed is required.  True if the action would be allowed, false otherwise.
	Allowed bool `json:"allowed"`
	// Reason is optional.  It indicates why a request was allowed or denied.
	Reason string `json:"reason,omitempty"`
}
