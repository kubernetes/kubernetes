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
type FakeClients struct {
	Fake *Fake
}

func (c *FakeClients) List(selector labels.Selector) (*oapi.OAuthClientList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-clients"})
	return api.Scheme.CopyOrDie(&c.Fake.ClientList).(*oapi.OAuthClientList), c.Fake.Err
}

func (c *FakeClients) Get(name string) (*oapi.OAuthClient, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-client", Value: name})
	return api.Scheme.CopyOrDie(&c.Fake.Client).(*oapi.OAuthClient), c.Fake.ClientGetErr
}

func (c *FakeClients) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-client", Value: name})
	return c.Fake.Err
}

func (c *FakeClients) Create(oauthclient *oapi.OAuthClient) (*oapi.OAuthClient, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-client", Value: oauthclient.Name})
	return &oapi.OAuthClient{}, c.Fake.Err
}

func (c *FakeClients) Update(oauthclient *oapi.OAuthClient) (*oapi.OAuthClient, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-client", Value: oauthclient.Name})
	return &oapi.OAuthClient{}, c.Fake.Err
}
