/*
Copyright 2014 Google Inc. All rights reserved.

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

package v1beta2

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
)

type OAuthAccessToken struct {
	v1beta2.TypeMeta `json:",inline"`
	Labels           map[string]string `json:"labels,omitempty"`

	// ClientName references the client that created this token.
	ClientName string `json:"clientName,omitempty"`

	// ExpiresIn is the seconds from CreationTime before this token expires.
	ExpiresIn int64 `json:"expiresIn,omitempty"`

	// Scopes is an array of the requested scopes.
	Scopes []string `json:"scopes,omitempty"`

	// RedirectURI is the redirection associated with the token.
	RedirectURI string `json:"redirectURI,omitempty"`

	// UserName is the user name associated with this token
	UserName string `json:"userName,omitempty"`

	// UserUID is the unique UID associated with this token
	UserUID string `json:"userUID,omitempty"`

	// AuthorizeToken contains information about the original authorization for this token
	AuthorizeToken OAuthAuthorizeToken `json:"authorizeToken,omitempty"`

	// RefreshToken is the value by which this token can be renewed. Can be blank.
	RefreshToken string `json:"refreshToken,omitempty"`
}

type OAuthAuthorizeToken struct {
	v1beta2.TypeMeta `json:",inline"`
	Labels           map[string]string `json:"labels,omitempty"`

	// ClientName references the client that created this token.
	ClientName string `json:"clientName,omitempty"`

	// ExpiresIn is the seconds from CreationTime before this token expires.
	ExpiresIn int64 `json:"expiresIn,omitempty"`

	// Scopes is an array of the requested scopes.
	Scopes []string `json:"scopes,omitempty"`

	// RedirectURI is the redirection associated with the token.
	RedirectURI string `json:"redirectURI,omitempty"`

	// State data from request
	State string `json:"state,omitempty"`

	// UserName is the user name associated with this token
	UserName string `json:"userName,omitempty"`

	// UserUID is the unique UID associated with this token
	UserUID string `json:"userUID,omitempty"`
}

type OAuthClient struct {
	v1beta2.TypeMeta `json:",inline"`
	Labels           map[string]string `json:"labels,omitempty"`

	// Secret is the unique secret associated with a client
	Secret string `json:"secret,omitempty"`

	// RedirectURIs is the valid redirection URIs associated with a client
	RedirectURIs []string `json:"redirectURIs,omitempty"`
}

type OAuthClientAuthorization struct {
	v1beta2.TypeMeta `json:",inline"`
	Labels           map[string]string `json:"labels,omitempty"`

	// ClientName references the client that created this authorization
	ClientName string `json:"clientName,omitempty"`

	// UserName is the user name that authorized this client
	UserName string `json:"userName,omitempty"`

	// UserUID is the unique UID associated with this authorization. UserUID and UserName
	// must both match for this authorization to be valid.
	UserUID string `json:"userUID,omitempty"`

	// Scopes is an array of the granted scopes.
	Scopes []string `json:"scopes,omitempty"`
}

type OAuthAccessTokenList struct {
	v1beta2.TypeMeta `json:",inline"`
	Items            []OAuthAccessToken `json:"items,omitempty"`
}

type OAuthAuthorizeTokenList struct {
	v1beta2.TypeMeta `json:",inline"`
	Items            []OAuthAuthorizeToken `json:"items,omitempty"`
}

type OAuthClientList struct {
	v1beta2.TypeMeta `json:",inline"`
	Items            []OAuthClient `json:"items,omitempty"`
}

type OAuthClientAuthorizationList struct {
	v1beta2.TypeMeta `json:",inline"`
	Items            []OAuthClientAuthorization `json:"items,omitempty"`
}
