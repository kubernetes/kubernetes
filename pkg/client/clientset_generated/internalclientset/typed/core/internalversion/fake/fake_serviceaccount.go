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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	api "k8s.io/kubernetes/pkg/api"
)

// FakeServiceAccounts implements ServiceAccountInterface
type FakeServiceAccounts struct {
	Fake *FakeCore
	ns   string
}

var serviceaccountsResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "serviceaccounts"}

var serviceaccountsKind = schema.GroupVersionKind{Group: "", Version: "", Kind: "ServiceAccount"}

func (c *FakeServiceAccounts) Create(serviceAccount *api.ServiceAccount) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(serviceaccountsResource, c.ns, serviceAccount), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Update(serviceAccount *api.ServiceAccount) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(serviceaccountsResource, c.ns, serviceAccount), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(serviceaccountsResource, c.ns, name), &api.ServiceAccount{})

	return err
}

func (c *FakeServiceAccounts) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(serviceaccountsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.ServiceAccountList{})
	return err
}

func (c *FakeServiceAccounts) Get(name string, options v1.GetOptions) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(serviceaccountsResource, c.ns, name), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}

func (c *FakeServiceAccounts) List(opts v1.ListOptions) (result *api.ServiceAccountList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(serviceaccountsResource, serviceaccountsKind, c.ns, opts), &api.ServiceAccountList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
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
func (c *FakeServiceAccounts) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(serviceaccountsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched serviceAccount.
func (c *FakeServiceAccounts) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.ServiceAccount, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(serviceaccountsResource, c.ns, name, data, subresources...), &api.ServiceAccount{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.ServiceAccount), err
}
