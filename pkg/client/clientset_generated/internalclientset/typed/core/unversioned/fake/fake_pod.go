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

// FakePods implements PodInterface
type FakePods struct {
	Fake *FakeCore
	ns   string
}

var podsResource = unversioned.GroupVersionResource{Group: "", Version: "", Resource: "pods"}

func (c *FakePods) Create(pod *api.Pod) (result *api.Pod, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(podsResource, c.ns, pod), &api.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Pod), err
}

func (c *FakePods) Update(pod *api.Pod) (result *api.Pod, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(podsResource, c.ns, pod), &api.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Pod), err
}

func (c *FakePods) UpdateStatus(pod *api.Pod) (*api.Pod, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(podsResource, "status", c.ns, pod), &api.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Pod), err
}

func (c *FakePods) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(podsResource, c.ns, name), &api.Pod{})

	return err
}

func (c *FakePods) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(podsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.PodList{})
	return err
}

func (c *FakePods) Get(name string) (result *api.Pod, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(podsResource, c.ns, name), &api.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Pod), err
}

func (c *FakePods) List(opts api.ListOptions) (result *api.PodList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(podsResource, c.ns, opts), &api.PodList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &api.PodList{}
	for _, item := range obj.(*api.PodList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested pods.
func (c *FakePods) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(podsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched pod.
func (c *FakePods) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.Pod, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(podsResource, c.ns, name, data, subresources...), &api.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.Pod), err
}
