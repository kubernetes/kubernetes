/*
Copyright 2017 The Kubernetes Authors.

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

// FakeServiceAccounts implements ServiceAccountInterface
type FakeServiceAccounts struct {
	Fake *FakeCore
	ns   string
}

var serviceaccountsResource = unversioned.GroupVersionResource{Group: "", Version: "", Resource: "serviceaccounts"}

func (c *FakeServiceAccounts) Create(serviceAccount *api.ServiceAccount) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(serviceaccountsResource, c.ns, serviceAccount), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Update(serviceAccount *api.ServiceAccount) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(serviceaccountsResource, c.ns, serviceAccount), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(serviceaccountsResource, c.ns, name), &api.ServiceAccount{})

	return err
}

func (c *FakeServiceAccounts) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(serviceaccountsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.ServiceAccountList{})
	return err
}

func (c *FakeServiceAccounts) Get(name string) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(serviceaccountsResource, c.ns, name), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) List(opts api.ListOptions) (result *api.ServiceAccountList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(serviceaccountsResource, c.ns, opts), &api.ServiceAccountList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.ServiceAccountList{}
	for _, item := range obj.(*api.ServiceAccountList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested serviceAccounts.
func (c *FakeServiceAccounts) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(serviceaccountsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched serviceAccount.
func (c *FakeServiceAccounts) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(serviceaccountsResource, c.ns, name, data, subresources...), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}
