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
	api "k8s.io/client-go/1.5/pkg/api"
	unversioned "k8s.io/client-go/1.5/pkg/api/unversioned"
	v1 "k8s.io/client-go/1.5/pkg/api/v1"
	labels "k8s.io/client-go/1.5/pkg/labels"
	watch "k8s.io/client-go/1.5/pkg/watch"
	testing "k8s.io/client-go/1.5/testing"
)

// FakePods implements PodInterface
type FakePods struct {
	Fake *FakeCore
	ns   string
}

var podsResource = unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}

func (c *FakePods) Create(pod *v1.Pod) (result *v1.Pod, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(podsResource, c.ns, pod), &v1.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Pod), err
}

func (c *FakePods) Update(pod *v1.Pod) (result *v1.Pod, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(podsResource, c.ns, pod), &v1.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Pod), err
}

func (c *FakePods) UpdateStatus(pod *v1.Pod) (*v1.Pod, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(podsResource, "status", c.ns, pod), &v1.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Pod), err
}

func (c *FakePods) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(podsResource, c.ns, name), &v1.Pod{})

	return err
}

func (c *FakePods) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := testing.NewDeleteCollectionAction(podsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.PodList{})
	return err
}

func (c *FakePods) Get(name string) (result *v1.Pod, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(podsResource, c.ns, name), &v1.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Pod), err
}

func (c *FakePods) List(opts api.ListOptions) (result *v1.PodList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(podsResource, c.ns, opts), &v1.PodList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.PodList{}
	for _, item := range obj.(*v1.PodList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested pods.
func (c *FakePods) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(podsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched pod.
func (c *FakePods) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Pod, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(podsResource, c.ns, name, data, subresources...), &v1.Pod{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Pod), err
}
