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
	apps "k8s.io/kubernetes/pkg/apis/apps"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeStatefulSets implements StatefulSetInterface
type FakeStatefulSets struct {
	Fake *FakeApps
	ns   string
}

var statefulsetsResource = unversioned.GroupVersionResource{Group: "apps", Version: "", Resource: "statefulsets"}

func (c *FakeStatefulSets) Create(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(statefulsetsResource, c.ns, statefulSet), &apps.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), err
}

func (c *FakeStatefulSets) Update(statefulSet *apps.StatefulSet) (result *apps.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(statefulsetsResource, c.ns, statefulSet), &apps.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), err
}

func (c *FakeStatefulSets) UpdateStatus(statefulSet *apps.StatefulSet) (*apps.StatefulSet, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(statefulsetsResource, "status", c.ns, statefulSet), &apps.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), err
}

func (c *FakeStatefulSets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(statefulsetsResource, c.ns, name), &apps.StatefulSet{})

	return err
}

func (c *FakeStatefulSets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(statefulsetsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &apps.StatefulSetList{})
	return err
}

func (c *FakeStatefulSets) Get(name string) (result *apps.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(statefulsetsResource, c.ns, name), &apps.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), err
}

func (c *FakeStatefulSets) List(opts api.ListOptions) (result *apps.StatefulSetList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(statefulsetsResource, c.ns, opts), &apps.StatefulSetList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &apps.StatefulSetList{}
	for _, item := range obj.(*apps.StatefulSetList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested statefulSets.
func (c *FakeStatefulSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(statefulsetsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched statefulSet.
func (c *FakeStatefulSets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apps.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(statefulsetsResource, c.ns, name, data, subresources...), &apps.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*apps.StatefulSet), err
}
