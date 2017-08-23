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

package authorization

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +genclient:noVerbs
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SubjectAccessReview checks whether or not a user or group can perform an action.  Not filling in a
// spec.namespace means "in all namespaces".
type SubjectAccessReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated
	Spec SubjectAccessReviewSpec

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus
}

// +genclient
// +genclient:nonNamespaced
// +genclient:noVerbs
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SelfSubjectAccessReview checks whether or the current user can perform an action.  Not filling in a
// spec.namespace means "in all namespaces".  Self is a special case, because users should always be able
// to check whether they can perform an action
type SelfSubjectAccessReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated.
	Spec SelfSubjectAccessReviewSpec

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus
}

// +genclient
// +genclient:noVerbs
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LocalSubjectAccessReview checks whether or not a user or group can perform an action in a given namespace.
// Having a namespace scoped resource makes it much easier to grant namespace scoped policy that includes permissions
// checking.
type LocalSubjectAccessReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated.  spec.namespace must be equal to the namespace
	// you made the request against.  If empty, it is defaulted.
	Spec SubjectAccessReviewSpec

	// Status is filled in by the server and indicates whether the request is allowed or not
	Status SubjectAccessReviewStatus
}

// ResourceAttributes includes the authorization attributes available for resource requests to the Authorizer interface
type ResourceAttributes struct {
	// Namespace is the namespace of the action being requested.  Currently, there is no distinction between no namespace and all namespaces
	// "" (empty) is defaulted for LocalSubjectAccessReviews
	// "" (empty) is empty for cluster-scoped resources
	// "" (empty) means "all" for namespace scoped resources from a SubjectAccessReview or SelfSubjectAccessReview
	Namespace string
	// Verb is a kubernetes resource API verb, like: get, list, watch, create, update, delete, proxy.  "*" means all.
	Verb string
	// Group is the API Group of the Resource.  "*" means all.
	Group string
	// Version is the API Version of the Resource.  "*" means all.
	Version string
	// Resource is one of the existing resource types.  "*" means all.
	Resource string
	// Subresource is one of the existing resource types.  "" means none.
	Subresource string
	// Name is the name of the resource being requested for a "get" or deleted for a "delete". "" (empty) means all.
	Name string
}

// NonResourceAttributes includes the authorization attributes available for non-resource requests to the Authorizer interface
type NonResourceAttributes struct {
	// Path is the URL path of the request
	Path string
	// Verb is the standard HTTP verb
	Verb string
}

// SubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAttributes
// and NonResourceAttributes must be set
type SubjectAccessReviewSpec struct {
	// ResourceAttributes describes information for a resource access request
	ResourceAttributes *ResourceAttributes
	// NonResourceAttributes describes information for a non-resource access request
	NonResourceAttributes *NonResourceAttributes

	// User is the user you're testing for.
	// If you specify "User" but not "Group", then is it interpreted as "What if User were not a member of any groups
	User string
	// Groups is the groups you're testing for.
	Groups []string
	// Extra corresponds to the user.Info.GetExtra() method from the authenticator.  Since that is input to the authorizer
	// it needs a reflection here.
	Extra map[string]ExtraValue
	// UID information about the requesting user.
	UID string
}

// ExtraValue masks the value so protobuf can generate
// +protobuf.nullable=true
type ExtraValue []string

// SelfSubjectAccessReviewSpec is a description of the access request.  Exactly one of ResourceAttributes
// and NonResourceAttributes must be set
type SelfSubjectAccessReviewSpec struct {
	// ResourceAttributes describes information for a resource access request
	ResourceAttributes *ResourceAttributes
	// NonResourceAttributes describes information for a non-resource access request
	NonResourceAttributes *NonResourceAttributes
}

// SubjectAccessReviewStatus
type SubjectAccessReviewStatus struct {
	// Allowed is required.  True if the action would be allowed, false otherwise.
	Allowed bool
	// Reason is optional.  It indicates why a request was allowed or denied.
	Reason string
	// EvaluationError is an indication that some error occurred during the authorization check.
	// It is entirely possible to get an error and be able to continue determine authorization status in spite of it.
	// For instance, RBAC can be missing a role, but enough roles are still present and bound to reason about the request.
	EvaluationError string
}
