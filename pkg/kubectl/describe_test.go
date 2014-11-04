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

package kubectl

import (
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

type describeClient struct {
	T         *testing.T
	Namespace string
	Err       error
	Fake      *client.Fake
}

func (c *describeClient) Pod(namespace string) (client.PodInterface, error) {
	if namespace != c.Namespace {
		c.T.Errorf("unexpected namespace arg: %s", namespace)
	}
	return c.Fake.Pods(namespace), c.Err
}

func (c *describeClient) ReplicationController(namespace string) (client.ReplicationControllerInterface, error) {
	if namespace != c.Namespace {
		c.T.Errorf("unexpected namespace arg: %s", namespace)
	}
	return c.Fake.ReplicationControllers(namespace), c.Err
}

func (c *describeClient) Service(namespace string) (client.ServiceInterface, error) {
	if namespace != c.Namespace {
		c.T.Errorf("unexpected namespace arg: %s", namespace)
	}
	return c.Fake.Services(namespace), c.Err
}

func TestDescribePod(t *testing.T) {
	fake := &client.Fake{}
	c := &describeClient{T: t, Namespace: "foo", Fake: fake}
	d := PodDescriber{
		PodClient:                   c.Pod,
		ReplicationControllerClient: c.ReplicationController,
	}
	out, err := d.Describe("foo", "bar")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "Status:") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeService(t *testing.T) {
	fake := &client.Fake{}
	c := &describeClient{T: t, Namespace: "foo", Fake: fake}
	d := ServiceDescriber{
		ServiceClient: c.Service,
	}
	out, err := d.Describe("foo", "bar")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "Labels:") || !strings.Contains(out, "bar") {
		t.Errorf("unexpected out: %s", out)
	}
}
