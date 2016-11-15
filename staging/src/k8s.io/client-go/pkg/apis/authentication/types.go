/*
Copyright 2016 The Kubernetes Authors.

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

package authentication

import (
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/unversioned"
)

const (
	// ImpersonateUserHeader is used to impersonate a particular user during an API server request
	ImpersonateUserHeader = "Impersonate-User"

	// ImpersonateGroupHeader is used to impersonate a particular group during an API server request.
	// It can be repeated multiplied times for multiple groups.
	ImpersonateGroupHeader = "Impersonate-Group"

	// ImpersonateUserExtraHeaderPrefix is a prefix for any header used to impersonate an entry in the
	// extra map[string][]string for user.Info.  The key will be every after the prefix.
	// It can be repeated multiplied times for multiple map keys and the same key can be repeated multiple
	// times to have multiple elements in the slice under a single key
	ImpersonateUserExtraHeaderPrefix = "Impersonate-Extra-"
)

// +genclient=true
// +nonNamespaced=true
// +noMethods=true

// TokenReview attempts to authenticate a token to a known user.
type TokenReview struct {
	unversioned.TypeMeta
	// ObjectMeta fulfills the meta.ObjectMetaAccessor interface so that the stock
	// REST handler paths work
	api.ObjectMeta

	// Spec holds information about the request being evaluated
	Spec TokenReviewSpec

	// Status is filled in by the server and indicates whether the request can be authenticated.
	Status TokenReviewStatus
}

// TokenReviewSpec is a description of the token authentication request.
type TokenReviewSpec struct {
	// Token is the opaque bearer token.
	Token string
}

// TokenReviewStatus is the result of the token authentication request.
// This type mirrors the authentication.Token interface
type TokenReviewStatus struct {
	// Authenticated indicates that the token was associated with a known user.
	Authenticated bool
	// User is the UserInfo associated with the provided token.
	User UserInfo
	// Error indicates that the token couldn't be checked
	Error string
}

// UserInfo holds the information about the user needed to implement the
// user.Info interface.
type UserInfo struct {
	// The name that uniquely identifies this user among all active users.
	Username string
	// A unique value that identifies this user across time. If this user is
	// deleted and another user by the same name is added, they will have
	// different UIDs.
	UID string
	// The names of groups this user is a part of.
	Groups []string
	// Any additional information provided by the authenticator.
	Extra map[string]ExtraValue
}

// ExtraValue masks the value so protobuf can generate
type ExtraValue []string
