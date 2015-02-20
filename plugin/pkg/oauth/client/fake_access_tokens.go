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
type FakeAccessTokens struct {
	Fake *Fake
}

func (c *FakeAccessTokens) List(selector labels.Selector) (*oapi.OAuthAccessTokenList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-access-tokens"})
	return api.Scheme.CopyOrDie(&c.Fake.AccessTokenList).(*oapi.OAuthAccessTokenList), c.Fake.Err
}

func (c *FakeAccessTokens) Get(name string) (*oapi.OAuthAccessToken, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-access-token", Value: name})
	return api.Scheme.CopyOrDie(&c.Fake.AccessToken).(*oapi.OAuthAccessToken), c.Fake.Err
}

func (c *FakeAccessTokens) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-access-token", Value: name})
	return c.Fake.Err
}

func (c *FakeAccessTokens) Create(token *oapi.OAuthAccessToken) (*oapi.OAuthAccessToken, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-access-token", Value: token.Name})
	return &oapi.OAuthAccessToken{}, c.Fake.Err
}

func (c *FakeAccessTokens) Update(token *oapi.OAuthAccessToken) (*oapi.OAuthAccessToken, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-access-token", Value: token.Name})
	return &oapi.OAuthAccessToken{}, c.Fake.Err
}
