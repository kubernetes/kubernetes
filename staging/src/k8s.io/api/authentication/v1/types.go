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

package v1

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	// ImpersonateUserHeader is used to impersonate a particular user during an API server request
	ImpersonateUserHeader = "Impersonate-User"

	// ImpersonateGroupHeader is used to impersonate a particular group during an API server request.
	// It can be repeated multiplied times for multiple groups.
	ImpersonateGroupHeader = "Impersonate-Group"

	// ImpersonateUIDHeader is used to impersonate a particular UID during an API server request
	ImpersonateUIDHeader = "Impersonate-Uid"

	// ImpersonateUserExtraHeaderPrefix is a prefix for any header used to impersonate an entry in the
	// extra map[string][]string for user.Info.  The key will be every after the prefix.
	// It can be repeated multiplied times for multiple map keys and the same key can be repeated multiple
	// times to have multiple elements in the slice under a single key
	ImpersonateUserExtraHeaderPrefix = "Impersonate-Extra-"
)

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TokenReview attempts to authenticate a token to a known user.
// Note: TokenReview requests may be cached by the webhook token authenticator
// plugin in the kube-apiserver.
type TokenReview struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec holds information about the request being evaluated
	Spec TokenReviewSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status is filled in by the server and indicates whether the request can be authenticated.
	// +optional
	Status TokenReviewStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// TokenReviewSpec is a description of the token authentication request.
type TokenReviewSpec struct {
	// Token is the opaque bearer token.
	// +optional
	Token string `json:"token,omitempty" protobuf:"bytes,1,opt,name=token"`
	// Audiences is a list of the identifiers that the resource server presented
	// with the token identifies as. Audience-aware token authenticators will
	// verify that the token was intended for at least one of the audiences in
	// this list. If no audiences are provided, the audience will default to the
	// audience of the Kubernetes apiserver.
	// +optional
	Audiences []string `json:"audiences,omitempty" protobuf:"bytes,2,rep,name=audiences"`
}

// TokenReviewStatus is the result of the token authentication request.
type TokenReviewStatus struct {
	// Authenticated indicates that the token was associated with a known user.
	// +optional
	Authenticated bool `json:"authenticated,omitempty" protobuf:"varint,1,opt,name=authenticated"`
	// User is the UserInfo associated with the provided token.
	// +optional
	User UserInfo `json:"user,omitempty" protobuf:"bytes,2,opt,name=user"`
	// Audiences are audience identifiers chosen by the authenticator that are
	// compatible with both the TokenReview and token. An identifier is any
	// identifier in the intersection of the TokenReviewSpec audiences and the
	// token's audiences. A client of the TokenReview API that sets the
	// spec.audiences field should validate that a compatible audience identifier
	// is returned in the status.audiences field to ensure that the TokenReview
	// server is audience aware. If a TokenReview returns an empty
	// status.audience field where status.authenticated is "true", the token is
	// valid against the audience of the Kubernetes API server.
	// +optional
	Audiences []string `json:"audiences,omitempty" protobuf:"bytes,4,rep,name=audiences"`
	// Error indicates that the token couldn't be checked
	// +optional
	Error string `json:"error,omitempty" protobuf:"bytes,3,opt,name=error"`
}

// UserInfo holds the information about the user needed to implement the
// user.Info interface.
type UserInfo struct {
	// The name that uniquely identifies this user among all active users.
	// +optional
	Username string `json:"username,omitempty" protobuf:"bytes,1,opt,name=username"`
	// A unique value that identifies this user across time. If this user is
	// deleted and another user by the same name is added, they will have
	// different UIDs.
	// +optional
	UID string `json:"uid,omitempty" protobuf:"bytes,2,opt,name=uid"`
	// The names of groups this user is a part of.
	// +optional
	Groups []string `json:"groups,omitempty" protobuf:"bytes,3,rep,name=groups"`
	// Any additional information provided by the authenticator.
	// +optional
	Extra map[string]ExtraValue `json:"extra,omitempty" protobuf:"bytes,4,rep,name=extra"`
}

// ExtraValue masks the value so protobuf can generate
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type ExtraValue []string

func (t ExtraValue) String() string {
	return fmt.Sprintf("%v", []string(t))
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TokenRequest requests a token for a given service account.
type TokenRequest struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec holds information about the request being evaluated
	Spec TokenRequestSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status is filled in by the server and indicates whether the token can be authenticated.
	// +optional
	Status TokenRequestStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// TokenRequestSpec contains client provided parameters of a token request.
type TokenRequestSpec struct {
	// Audiences are the intendend audiences of the token. A recipient of a
	// token must identitfy themself with an identifier in the list of
	// audiences of the token, and otherwise should reject the token. A
	// token issued for multiple audiences may be used to authenticate
	// against any of the audiences listed but implies a high degree of
	// trust between the target audiences.
	Audiences []string `json:"audiences" protobuf:"bytes,1,rep,name=audiences"`

	// ExpirationSeconds is the requested duration of validity of the request. The
	// token issuer may return a token with a different validity duration so a
	// client needs to check the 'expiration' field in a response.
	// +optional
	ExpirationSeconds *int64 `json:"expirationSeconds" protobuf:"varint,4,opt,name=expirationSeconds"`

	// BoundObjectRef is a reference to an object that the token will be bound to.
	// The token will only be valid for as long as the bound object exists.
	// NOTE: The API server's TokenReview endpoint will validate the
	// BoundObjectRef, but other audiences may not. Keep ExpirationSeconds
	// small if you want prompt revocation.
	// +optional
	BoundObjectRef *BoundObjectReference `json:"boundObjectRef" protobuf:"bytes,3,opt,name=boundObjectRef"`
}

// TokenRequestStatus is the result of a token request.
type TokenRequestStatus struct {
	// Token is the opaque bearer token.
	Token string `json:"token" protobuf:"bytes,1,opt,name=token"`
	// ExpirationTimestamp is the time of expiration of the returned token.
	ExpirationTimestamp metav1.Time `json:"expirationTimestamp" protobuf:"bytes,2,opt,name=expirationTimestamp"`
}

// BoundObjectReference is a reference to an object that a token is bound to.
type BoundObjectReference struct {
	// Kind of the referent. Valid kinds are 'Pod' and 'Secret'.
	// +optional
	Kind string `json:"kind,omitempty" protobuf:"bytes,1,opt,name=kind"`
	// API version of the referent.
	// +optional
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,2,opt,name=apiVersion"`

	// Name of the referent.
	// +optional
	Name string `json:"name,omitempty" protobuf:"bytes,3,opt,name=name"`
	// UID of the referent.
	// +optional
	UID types.UID `json:"uid,omitempty" protobuf:"bytes,4,opt,name=uID,casttype=k8s.io/apimachinery/pkg/types.UID"`
}
