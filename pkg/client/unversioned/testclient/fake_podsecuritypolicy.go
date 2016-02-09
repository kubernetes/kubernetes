/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

// FakePodSecurityPolicy implements PodSecurityPolicyInterface. Meant to be
// embedded into a struct to get a default implementation. This makes faking out just
// the method you want to test easier.
type FakePodSecurityPolicy struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePodSecurityPolicy) List(opts api.ListOptions) (*extensions.PodSecurityPolicyList, error) {
	obj, err := c.Fake.Invokes(NewListAction("podsecuritypolicies", c.Namespace, opts), &extensions.PodSecurityPolicyList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.PodSecurityPolicyList), err
}

func (c *FakePodSecurityPolicy) Get(name string) (*extensions.PodSecurityPolicy, error) {
	obj, err := c.Fake.Invokes(NewGetAction("podsecuritypolicies", c.Namespace, name), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicy) Create(scc *extensions.PodSecurityPolicy) (*extensions.PodSecurityPolicy, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("podsecuritypolicies", c.Namespace, scc), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicy) Update(scc *extensions.PodSecurityPolicy) (*extensions.PodSecurityPolicy, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("podsecuritypolicies", c.Namespace, scc), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicy) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("podsecuritypolicies", c.Namespace, name), &extensions.PodSecurityPolicy{})
	return err
}

func (c *FakePodSecurityPolicy) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("podsecuritypolicies", c.Namespace, opts))
}
