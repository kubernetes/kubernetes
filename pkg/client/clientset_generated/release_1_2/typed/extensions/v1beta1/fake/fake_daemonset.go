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
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeDaemonSets implements DaemonSetInterface
type FakeDaemonSets struct {
	Fake *FakeExtensions
	ns   string
}

var daemonsetsResource = unversioned.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "daemonsets"}

func (c *FakeDaemonSets) Create(daemonSet *v1beta1.DaemonSet) (result *v1beta1.DaemonSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(daemonsetsResource, c.ns, daemonSet), &v1beta1.DaemonSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.DaemonSet), err
}

func (c *FakeDaemonSets) Update(daemonSet *v1beta1.DaemonSet) (result *v1beta1.DaemonSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(daemonsetsResource, c.ns, daemonSet), &v1beta1.DaemonSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.DaemonSet), err
}

func (c *FakeDaemonSets) UpdateStatus(daemonSet *v1beta1.DaemonSet) (*v1beta1.DaemonSet, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(daemonsetsResource, "status", c.ns, daemonSet), &v1beta1.DaemonSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.DaemonSet), err
}

func (c *FakeDaemonSets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(daemonsetsResource, c.ns, name), &v1beta1.DaemonSet{})

	return err
}

func (c *FakeDaemonSets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(daemonsetsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.DaemonSetList{})
	return err
}

func (c *FakeDaemonSets) Get(name string) (result *v1beta1.DaemonSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(daemonsetsResource, c.ns, name), &v1beta1.DaemonSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.DaemonSet), err
}

func (c *FakeDaemonSets) List(opts api.ListOptions) (result *v1beta1.DaemonSetList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(daemonsetsResource, c.ns, opts), &v1beta1.DaemonSetList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.DaemonSetList{}
	for _, item := range obj.(*v1beta1.DaemonSetList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested daemonSets.
func (c *FakeDaemonSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(daemonsetsResource, c.ns, opts))

}
