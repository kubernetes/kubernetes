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
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeSecurityContextConstraints implements SecurityContextConstraintsInterface
type FakeSecurityContextConstraints struct {
	Fake *FakeCore
}

var securitycontextconstraintsResource = unversioned.GroupVersionResource{Group: "", Version: "", Resource: "securitycontextconstraints"}

func (c *FakeSecurityContextConstraints) Create(securityContextConstraints *api.SecurityContextConstraints) (result *api.SecurityContextConstraints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(securitycontextconstraintsResource, securityContextConstraints), &api.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) Update(securityContextConstraints *api.SecurityContextConstraints) (result *api.SecurityContextConstraints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(securitycontextconstraintsResource, securityContextConstraints), &api.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(securitycontextconstraintsResource, name), &api.SecurityContextConstraints{})
	return err
}

func (c *FakeSecurityContextConstraints) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(securitycontextconstraintsResource, listOptions)

	_, err := c.Fake.Invokes(action, &api.SecurityContextConstraintsList{})
	return err
}

func (c *FakeSecurityContextConstraints) Get(name string) (result *api.SecurityContextConstraints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(securitycontextconstraintsResource, name), &api.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), err
}

func (c *FakeSecurityContextConstraints) List(opts api.ListOptions) (result *api.SecurityContextConstraintsList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(securitycontextconstraintsResource, opts), &api.SecurityContextConstraintsList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &api.SecurityContextConstraintsList{}
	for _, item := range obj.(*api.SecurityContextConstraintsList).Items {
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

// Patch applies the patch and returns the patched securityContextConstraints.
func (c *FakeSecurityContextConstraints) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.SecurityContextConstraints, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(securitycontextconstraintsResource, name, data, subresources...), &api.SecurityContextConstraints{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.SecurityContextConstraints), err
}
