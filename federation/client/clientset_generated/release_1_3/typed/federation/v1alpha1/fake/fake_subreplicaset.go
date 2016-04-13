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
	v1alpha1 "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	api "k8s.io/kubernetes/pkg/api"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeSubReplicaSets implements SubReplicaSetInterface
type FakeSubReplicaSets struct {
	Fake *FakeFederation
	ns   string
}

func (c *FakeSubReplicaSets) Create(subReplicaSet *v1alpha1.SubReplicaSet) (result *v1alpha1.SubReplicaSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction("subreplicasets", c.ns, subReplicaSet), &v1alpha1.SubReplicaSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.SubReplicaSet), err
}

func (c *FakeSubReplicaSets) Update(subReplicaSet *v1alpha1.SubReplicaSet) (result *v1alpha1.SubReplicaSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction("subreplicasets", c.ns, subReplicaSet), &v1alpha1.SubReplicaSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.SubReplicaSet), err
}

func (c *FakeSubReplicaSets) UpdateStatus(subReplicaSet *v1alpha1.SubReplicaSet) (*v1alpha1.SubReplicaSet, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction("subreplicasets", "status", c.ns, subReplicaSet), &v1alpha1.SubReplicaSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.SubReplicaSet), err
}

func (c *FakeSubReplicaSets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction("subreplicasets", c.ns, name), &v1alpha1.SubReplicaSet{})

	return err
}

func (c *FakeSubReplicaSets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction("subreplicasets", c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.SubReplicaSetList{})
	return err
}

func (c *FakeSubReplicaSets) Get(name string) (result *v1alpha1.SubReplicaSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction("subreplicasets", c.ns, name), &v1alpha1.SubReplicaSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.SubReplicaSet), err
}

func (c *FakeSubReplicaSets) List(opts api.ListOptions) (result *v1alpha1.SubReplicaSetList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction("subreplicasets", c.ns, opts), &v1alpha1.SubReplicaSetList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.SubReplicaSetList{}
	for _, item := range obj.(*v1alpha1.SubReplicaSetList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested subReplicaSets.
func (c *FakeSubReplicaSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction("subreplicasets", c.ns, opts))

}
