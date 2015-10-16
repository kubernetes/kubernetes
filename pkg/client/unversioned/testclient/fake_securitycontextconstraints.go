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
	"k8s.io/kubernetes/pkg/watch"
)

// FakeSecurityContextConstraints implements SecurityContextConstraintInterface. Meant to be
// embedded into a struct to get a default implementation. This makes faking out just
// the method you want to test easier.
type FakeSecurityContextConstraints struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeSecurityContextConstraints) List(opts api.ListOptions) (*api.SecurityContextConstraintsList, error) {
	obj, err := c.Fake.Invokes(NewListAction("securitycontextconstraints", c.Namespace, opts), &api.SecurityContextConstraintsList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraintsList), err
}

func (c *FakeSecurityContextConstraints) Get(name string) (*api.SecurityContextConstraints, error) {
	obj, err := c.Fake.Invokes(NewGetAction("securitycontextconstraints", c.Namespace, name), &api.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) Create(scc *api.SecurityContextConstraints) (*api.SecurityContextConstraints, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("securitycontextconstraints", c.Namespace, scc), &api.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) Update(scc *api.SecurityContextConstraints) (*api.SecurityContextConstraints, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("securitycontextconstraints", c.Namespace, scc), &api.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("securitycontextconstraints", c.Namespace, name), &api.SecurityContextConstraints{})
	return err
}

func (c *FakeSecurityContextConstraints) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("securitycontextconstraints", c.Namespace, opts))
}
