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

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
)

// FakePods implements PodsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeAuthorizeTokens struct {
	Fake *Fake
}

func (c *FakeAuthorizeTokens) List(selector labels.Selector) (*oapi.OAuthAuthorizeTokenList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-authorize-tokens"})
	return api.Scheme.CopyOrDie(&c.Fake.AuthorizeTokenList).(*oapi.OAuthAuthorizeTokenList), c.Fake.Err
}

func (c *FakeAuthorizeTokens) Get(name string) (*oapi.OAuthAuthorizeToken, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-authorize-token", Value: name})
	return api.Scheme.CopyOrDie(&c.Fake.AuthorizeToken).(*oapi.OAuthAuthorizeToken), c.Fake.Err
}

func (c *FakeAuthorizeTokens) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-authorize-token", Value: name})
	return c.Fake.Err
}

func (c *FakeAuthorizeTokens) Create(token *oapi.OAuthAuthorizeToken) (*oapi.OAuthAuthorizeToken, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-authorize-token", Value: token.Name})
	return &oapi.OAuthAuthorizeToken{}, c.Fake.Err
}

func (c *FakeAuthorizeTokens) Update(token *oapi.OAuthAuthorizeToken) (*oapi.OAuthAuthorizeToken, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-authorize-token", Value: token.Name})
	return &oapi.OAuthAuthorizeToken{}, c.Fake.Err
}
