/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// TokenReview attempts to authenticate a token to a known user.
// Note: TokenReview requests may be cached by the webhook token authenticator
// plugin in the kube-apiserver.
type TokenReview struct {
	unversioned.TypeMeta `json:",inline"`

	// Spec holds information about the request being evaluated
	Spec TokenReviewSpec `json:"spec"`

	// Status is filled in by the server and indicates whether the request can be authenticated.
	Status TokenReviewStatus `json:"status,omitempty"`
}

// TokenReviewSpec is a description of the token authentication request.
type TokenReviewSpec struct {
	// Token is the opaque bearer token.
	Token string `json:"token,omitempty"`
}

// TokenReviewStatus is the result of the token authentication request.
type TokenReviewStatus struct {
	// Authenticated indicates that the token was associated with a known user.
	Authenticated bool `json:"authenticated,omitempty"`
	// User is the UserInfo associated with the provided token.
	User UserInfo `json:"user,omitempty"`
}

// UserInfo holds the information about the user needed to implement the
// user.Info interface.
type UserInfo struct {
	// The name that uniquely identifies this user among all active users.
	Username string `json:"username,omitempty"`
	// A unique value that identifies this user across time. If this user is
	// deleted and another user by the same name is added, they will have
	// different UIDs.
	UID string `json:"uid,omitempty"`
	// The names of groups this user is a part of.
	Groups []string `json:"groups,omitempty"`
	// Any additional information provided by the authenticator.
	Extra map[string][]string `json:"extra,omitempty"`
}
