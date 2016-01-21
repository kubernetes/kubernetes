/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeIngress implements IngressInterface
type FakeIngress struct {
	Fake *FakeExtensions
	ns   string
}

func (c *FakeIngress) Create(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction("ingress", c.ns, ingress), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngress) Update(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction("ingress", c.ns, ingress), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngress) UpdateStatus(ingress *extensions.Ingress) (*extensions.Ingress, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction("ingress", "status", c.ns, ingress), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngress) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction("ingress", c.ns, name), &extensions.Ingress{})

	return err
}

func (c *FakeIngress) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction("events", c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.IngressList{})
	return err
}

func (c *FakeIngress) Get(name string) (result *extensions.Ingress, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction("ingress", c.ns, name), &extensions.Ingress{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.Ingress), err
}

func (c *FakeIngress) List(opts api.ListOptions) (result *extensions.IngressList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction("ingress", c.ns, opts), &extensions.IngressList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.IngressList{}
	for _, item := range obj.(*extensions.IngressList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested ingress.
func (c *FakeIngress) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction("ingress", c.ns, opts))

}
