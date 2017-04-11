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
	v1beta1 "k8s.io/client-go/pkg/apis/apps/v1beta1"
	testing "k8s.io/client-go/testing"
)

// FakeStatefulSets implements StatefulSetInterface
type FakeStatefulSets struct {
	Fake *FakeAppsV1beta1
	ns   string
}

var statefulsetsResource = schema.GroupVersionResource{Group: "apps", Version: "v1beta1", Resource: "statefulsets"}

func (c *FakeStatefulSets) Create(statefulSet *v1beta1.StatefulSet) (result *v1beta1.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(statefulsetsResource, c.ns, statefulSet), &v1beta1.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.StatefulSet), err
}

func (c *FakeStatefulSets) Update(statefulSet *v1beta1.StatefulSet) (result *v1beta1.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(statefulsetsResource, c.ns, statefulSet), &v1beta1.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.StatefulSet), err
}

func (c *FakeStatefulSets) UpdateStatus(statefulSet *v1beta1.StatefulSet) (*v1beta1.StatefulSet, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(statefulsetsResource, "status", c.ns, statefulSet), &v1beta1.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.StatefulSet), err
}

func (c *FakeStatefulSets) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(statefulsetsResource, c.ns, name), &v1beta1.StatefulSet{})

	return err
}

func (c *FakeStatefulSets) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(statefulsetsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.StatefulSetList{})
	return err
}

func (c *FakeStatefulSets) Get(name string, options v1.GetOptions) (result *v1beta1.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(statefulsetsResource, c.ns, name), &v1beta1.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.StatefulSet), err
}

func (c *FakeStatefulSets) List(opts v1.ListOptions) (result *v1beta1.StatefulSetList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(statefulsetsResource, c.ns, opts), &v1beta1.StatefulSetList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.StatefulSetList{}
	for _, item := range obj.(*v1beta1.StatefulSetList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested statefulSets.
func (c *FakeStatefulSets) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(statefulsetsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched statefulSet.
func (c *FakeStatefulSets) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.StatefulSet, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(statefulsetsResource, c.ns, name, data, subresources...), &v1beta1.StatefulSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.StatefulSet), err
}
