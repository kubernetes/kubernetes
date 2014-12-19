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

package client

import "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"

type FakeAction struct {
	Action string
	Value  interface{}
}

type Fake struct {
	Actions []FakeAction
	Err     error

	AccessToken        api.OAuthAccessToken
	AccessTokenList    api.OAuthAccessTokenList
	AuthorizeToken     api.OAuthAuthorizeToken
	AuthorizeTokenList api.OAuthAuthorizeTokenList

	Client                  api.OAuthClient
	ClientList              api.OAuthClientList
	ClientAuthorization     api.OAuthClientAuthorization
	ClientAuthorizationList api.OAuthClientAuthorizationList

	ClientGetErr              error
	ClientAuthorizationGetErr error
}

func (c *Fake) OAuthAccessTokens() OAuthAccessTokenInterface {
	return &FakeAccessTokens{Fake: c}
}
func (c *Fake) OAuthAuthorizeTokens() OAuthAuthorizeTokenInterface {
	return &FakeAuthorizeTokens{Fake: c}
}
func (c *Fake) OAuthClients() OAuthClientInterface {
	return &FakeClients{Fake: c}
}
func (c *Fake) OAuthClientAuthorizations() OAuthClientAuthorizationInterface {
	return &FakeClientAuthorizations{Fake: c}
}
