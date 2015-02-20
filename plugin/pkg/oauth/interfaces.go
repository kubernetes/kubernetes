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

package oauth

import "github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"

type ClientAuthenticator interface {
	AuthenticateClient(client Client) (user.Info, bool, error)
}

type Client interface {
	GetId() string
	GetSecret() string
	GetRedirectUri() string
	GetUserData() interface{}
}

type Grant interface {
	GetClient() Client
	GetScopes() []string
	GetExpiration() int64
	GetRedirectUri() string
}

type DefaultGrant struct {
	Client      Client
	Scopes      []string
	Expiration  int64
	RedirectURI string
}

func (g *DefaultGrant) GetClient() Client      { return g.Client }
func (g *DefaultGrant) GetScopes() []string    { return g.Scopes }
func (g *DefaultGrant) GetExpiration() int64   { return g.Expiration }
func (g *DefaultGrant) GetRedirectUri() string { return g.RedirectURI }
