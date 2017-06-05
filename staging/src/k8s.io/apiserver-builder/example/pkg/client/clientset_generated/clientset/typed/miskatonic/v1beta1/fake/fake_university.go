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
	v1beta1 "k8s.io/apiserver-builder/example/pkg/apis/miskatonic/v1beta1"
	testing "k8s.io/client-go/testing"
)

// FakeUniversities implements UniversityInterface
type FakeUniversities struct {
	Fake *FakeMiskatonicV1beta1
	ns   string
}

var universitiesResource = schema.GroupVersionResource{Group: "miskatonic.k8s.io", Version: "v1beta1", Resource: "universities"}

func (c *FakeUniversities) Create(university *v1beta1.University) (result *v1beta1.University, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(universitiesResource, c.ns, university), &v1beta1.University{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.University), err
}

func (c *FakeUniversities) Update(university *v1beta1.University) (result *v1beta1.University, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(universitiesResource, c.ns, university), &v1beta1.University{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.University), err
}

func (c *FakeUniversities) UpdateStatus(university *v1beta1.University) (*v1beta1.University, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(universitiesResource, "status", c.ns, university), &v1beta1.University{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.University), err
}

func (c *FakeUniversities) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(universitiesResource, c.ns, name), &v1beta1.University{})

	return err
}

func (c *FakeUniversities) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(universitiesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.UniversityList{})
	return err
}

func (c *FakeUniversities) Get(name string, options v1.GetOptions) (result *v1beta1.University, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(universitiesResource, c.ns, name), &v1beta1.University{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.University), err
}

func (c *FakeUniversities) List(opts v1.ListOptions) (result *v1beta1.UniversityList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(universitiesResource, c.ns, opts), &v1beta1.UniversityList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.UniversityList{}
	for _, item := range obj.(*v1beta1.UniversityList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested universities.
func (c *FakeUniversities) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(universitiesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched university.
func (c *FakeUniversities) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.University, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(universitiesResource, c.ns, name, data, subresources...), &v1beta1.University{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.University), err
}
