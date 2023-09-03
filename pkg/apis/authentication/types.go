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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	// ImpersonateUserHeader is used to impersonate a particular user during an API server request
	ImpersonateUserHeader = "Impersonate-User"

	// ImpersonateUIDHeader is used to impersonate a particular UID during an API server request.
	ImpersonateUIDHeader = "Impersonate-Uid"

	// ImpersonateGroupHeader is used to impersonate a particular group during an API server request.
	// It can be repeated multiplied times for multiple groups.
	ImpersonateGroupHeader = "Impersonate-Group"

	// ImpersonateUserExtraHeaderPrefix is a prefix for any header used to impersonate an entry in the
	// extra map[string][]string for user.Info.  The key will be every after the prefix.
	// It can be repeated multiplied times for multiple map keys and the same key can be repeated multiple
	// times to have multiple elements in the slice under a single key
	ImpersonateUserExtraHeaderPrefix = "Impersonate-Extra-"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TokenReview attempts to authenticate a token to a known user.
type TokenReview struct {
	metav1.TypeMeta
	// ObjectMeta fulfills the metav1.ObjectMetaAccessor interface so that the stock
	// REST handler paths work
	metav1.ObjectMeta

	// Spec holds information about the request being evaluated
	Spec TokenReviewSpec

	// Status is filled in by the server and indicates whether the request can be authenticated.
	Status TokenReviewStatus
}

// TokenReviewSpec is a description of the token authentication request.
type TokenReviewSpec struct {
	// Token is the opaque bearer token.
	Token string `datapolicy:"token"`
	// Audiences is a list of the identifiers that the resource server presented
	// with the token identifies as. Audience-aware token authenticators will
	// verify that the token was intended for at least one of the audiences in
	// this list. If no audiences are provided, the audience will default to the
	// audience of the Kubernetes apiserver.
	Audiences []string
}

// TokenReviewStatus is the result of the token authentication request.
// This type mirrors the authentication.Token interface
type TokenReviewStatus struct {
	// Authenticated indicates that the token was associated with a known user.
	Authenticated bool
	// User is the UserInfo associated with the provided token.
	User UserInfo
	// Audiences are audience identifiers chosen by the authenticator that are
	// compatible with both the TokenReview and token. An identifier is any
	// identifier in the intersection of the TokenReviewSpec audiences and the
	// token's audiences. A client of the TokenReview API that sets the
	// spec.audiences field should validate that a compatible audience identifier
	// is returned in the status.audiences field to ensure that the TokenReview
	// server is audience aware. If a TokenReview returns an empty
	// status.audience field where status.authenticated is "true", the token is
	// valid against the audience of the Kubernetes API server.
	Audiences []string
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TokenRequest requests a token for a given service account.
type TokenRequest struct {
	metav1.TypeMeta
	// ObjectMeta fulfills the metav1.ObjectMetaAccessor interface so that the stock
	// REST handler paths work
	metav1.ObjectMeta

	Spec   TokenRequestSpec
	Status TokenRequestStatus
}

// TokenRequestSpec contains client provided parameters of a token request.
type TokenRequestSpec struct {
	// Audiences are the intended audiences of the token. A recipient of a
	// token must identify themself with an identifier in the list of
	// audiences of the token, and otherwise should reject the token. A
	// token issued for multiple audiences may be used to authenticate
	// against any of the audiences listed but implies a high degree of
	// trust between the target audiences.
	Audiences []string

	// ExpirationSeconds is the requested duration of validity of the request. The
	// token issuer may return a token with a different validity duration so a
	// client needs to check the 'expiration' field in a response.
	ExpirationSeconds int64

	// BoundObjectRef is a reference to an object that the token will be bound to.
	// The token will only be valid for as long as the bound object exists.
	// NOTE: The API server's TokenReview endpoint will validate the
	// BoundObjectRef, but other audiences may not. Keep ExpirationSeconds
	// small if you want prompt revocation.
	BoundObjectRef *BoundObjectReference
}

// TokenRequestStatus is the result of a token request.
type TokenRequestStatus struct {
	// Token is the opaque bearer token.
	Token string `datapolicy:"token"`
	// ExpirationTimestamp is the time of expiration of the returned token.
	ExpirationTimestamp metav1.Time
}

// BoundObjectReference is a reference to an object that a token is bound to.
type BoundObjectReference struct {
	// Kind of the referent. Valid kinds are 'Pod' and 'Secret'.
	Kind string
	// API version of the referent.
	APIVersion string

	// Name of the referent.
	Name string
	// UID of the referent.
	UID types.UID
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SelfSubjectReview contains the user information that the kube-apiserver has about the user making this request.
// When using impersonation, users will receive the user info of the user being impersonated.  If impersonation or
// request header authentication is used, any extra keys will have their case ignored and returned as lowercase.
type SelfSubjectReview struct {
	metav1.TypeMeta
	// ObjectMeta fulfills the metav1.ObjectMetaAccessor interface so that the stock.
	// REST handler paths work.
	metav1.ObjectMeta
	// Status is filled in by the server with the user attributes.
	Status SelfSubjectReviewStatus
}

// SelfSubjectReviewStatus is filled by the kube-apiserver and sent back to a user.
type SelfSubjectReviewStatus struct {
	// User attributes of the user making this request.
	UserInfo UserInfo
}
