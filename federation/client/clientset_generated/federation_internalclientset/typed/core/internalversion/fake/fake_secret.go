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

// FakeSecrets implements SecretInterface
type FakeSecrets struct {
	Fake *FakeCore
	ns   string
}

var secretsResource = unversioned.GroupVersionResource{Group: "", Version: "", Resource: "secrets"}

func (c *FakeSecrets) Create(secret *api.Secret) (result *api.Secret, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(secretsResource, c.ns, secret), &api.Secret{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Secret), err
}

func (c *FakeSecrets) Update(secret *api.Secret) (result *api.Secret, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(secretsResource, c.ns, secret), &api.Secret{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Secret), err
}

func (c *FakeSecrets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(secretsResource, c.ns, name), &api.Secret{})

	return err
}

func (c *FakeSecrets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(secretsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.SecretList{})
	return err
}

func (c *FakeSecrets) Get(name string) (result *api.Secret, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(secretsResource, c.ns, name), &api.Secret{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Secret), err
}

func (c *FakeSecrets) List(opts api.ListOptions) (result *api.SecretList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(secretsResource, c.ns, opts), &api.SecretList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.SecretList{}
	for _, item := range obj.(*api.SecretList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested secrets.
func (c *FakeSecrets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(secretsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched secret.
func (c *FakeSecrets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.Secret, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(secretsResource, c.ns, name, data, subresources...), &api.Secret{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Secret), err
}
