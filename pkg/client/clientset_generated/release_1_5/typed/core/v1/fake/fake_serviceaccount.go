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
	v1 "k8s.io/kubernetes/pkg/api/v1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeServiceAccounts implements ServiceAccountInterface
type FakeServiceAccounts struct {
	Fake *FakeCoreV1
	ns   string
}

var serviceaccountsResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"}

func (c *FakeServiceAccounts) Create(serviceAccount *v1.ServiceAccount) (result *v1.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(serviceaccountsResource, c.ns, serviceAccount), &v1.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ServiceAccount), err
}

func (c *FakeServiceAccounts) Update(serviceAccount *v1.ServiceAccount) (result *v1.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(serviceaccountsResource, c.ns, serviceAccount), &v1.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ServiceAccount), err
}

func (c *FakeServiceAccounts) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(serviceaccountsResource, c.ns, name), &v1.ServiceAccount{})

	return err
}

func (c *FakeServiceAccounts) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewDeleteCollectionAction(serviceaccountsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.ServiceAccountList{})
	return err
}

func (c *FakeServiceAccounts) Get(name string) (result *v1.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(serviceaccountsResource, c.ns, name), &v1.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ServiceAccount), err
}

func (c *FakeServiceAccounts) List(opts v1.ListOptions) (result *v1.ServiceAccountList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(serviceaccountsResource, c.ns, opts), &v1.ServiceAccountList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.ServiceAccountList{}
	for _, item := range obj.(*v1.ServiceAccountList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested serviceAccounts.
func (c *FakeServiceAccounts) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(serviceaccountsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched serviceAccount.
func (c *FakeServiceAccounts) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(serviceaccountsResource, c.ns, name, data, subresources...), &v1.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.ServiceAccount), err
}
