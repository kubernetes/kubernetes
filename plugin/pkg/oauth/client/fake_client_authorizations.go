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
type FakeClientAuthorizations struct {
	Fake *Fake
}

func (c *FakeClientAuthorizations) Name(username, clientname string) string {
	return username + ":" + clientname
}

func (c *FakeClientAuthorizations) List(selector labels.Selector) (*oapi.OAuthClientAuthorizationList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-client-authorizations"})
	return api.Scheme.CopyOrDie(&c.Fake.ClientAuthorizationList).(*oapi.OAuthClientAuthorizationList), c.Fake.Err
}

func (c *FakeClientAuthorizations) Get(name string) (*oapi.OAuthClientAuthorization, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-client-authorization", Value: name})
	return api.Scheme.CopyOrDie(&c.Fake.ClientAuthorization).(*oapi.OAuthClientAuthorization), c.Fake.ClientAuthorizationGetErr
}

func (c *FakeClientAuthorizations) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-client-authorization", Value: name})
	return c.Fake.Err
}

func (c *FakeClientAuthorizations) Create(auth *oapi.OAuthClientAuthorization) (*oapi.OAuthClientAuthorization, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-client-authorization", Value: auth})
	return &oapi.OAuthClientAuthorization{}, c.Fake.Err
}

func (c *FakeClientAuthorizations) Update(auth *oapi.OAuthClientAuthorization) (*oapi.OAuthClientAuthorization, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-client-authorization", Value: auth})
	return &oapi.OAuthClientAuthorization{}, c.Fake.Err
}
