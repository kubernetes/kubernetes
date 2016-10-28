/*
Copyright 2016 The Kubernetes Authors.

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

package fake

import (
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeSecurityContextConstraints implements SecurityContextConstraintsInterface
type FakeSecurityContextConstraints struct {
	Fake *FakeCore
}

var securitycontextconstraintsResource = unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "securitycontextconstraints"}

func (c *FakeSecurityContextConstraints) Create(securityContextConstraints *v1.SecurityContextConstraints) (result *v1.SecurityContextConstraints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(securitycontextconstraintsResource, securityContextConstraints), &v1.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) Update(securityContextConstraints *v1.SecurityContextConstraints) (result *v1.SecurityContextConstraints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(securitycontextconstraintsResource, securityContextConstraints), &v1.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(securitycontextconstraintsResource, name), &v1.SecurityContextConstraints{})
	return err
}

func (c *FakeSecurityContextConstraints) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(securitycontextconstraintsResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1.SecurityContextConstraintsList{})
	return err
}

func (c *FakeSecurityContextConstraints) Get(name string) (result *v1.SecurityContextConstraints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(securitycontextconstraintsResource, name), &v1.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) List(opts api.ListOptions) (result *v1.SecurityContextConstraintsList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(securitycontextconstraintsResource, opts), &v1.SecurityContextConstraintsList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.SecurityContextConstraintsList{}
	for _, item := range obj.(*v1.SecurityContextConstraintsList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested securityContextConstraints.
func (c *FakeSecurityContextConstraints) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(securitycontextconstraintsResource, opts))
}
