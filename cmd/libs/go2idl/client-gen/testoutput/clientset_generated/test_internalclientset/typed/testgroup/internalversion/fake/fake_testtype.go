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
	testgroup "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup"
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeTestTypes implements TestTypeInterface
type FakeTestTypes struct {
	Fake *FakeTestgroup
	ns   string
}

var testtypesResource = unversioned.GroupVersionResource{Group: "testgroup.k8s.io", Version: "", Resource: "testtypes"}

func (c *FakeTestTypes) Create(testType *testgroup.TestType) (result *testgroup.TestType, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(testtypesResource, c.ns, testType), &testgroup.TestType{})

	if obj == nil {
		return nil, err
	}
	return obj.(*testgroup.TestType), err
}

func (c *FakeTestTypes) Update(testType *testgroup.TestType) (result *testgroup.TestType, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(testtypesResource, c.ns, testType), &testgroup.TestType{})

	if obj == nil {
		return nil, err
	}
	return obj.(*testgroup.TestType), err
}

func (c *FakeTestTypes) UpdateStatus(testType *testgroup.TestType) (*testgroup.TestType, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(testtypesResource, "status", c.ns, testType), &testgroup.TestType{})

	if obj == nil {
		return nil, err
	}
	return obj.(*testgroup.TestType), err
}

func (c *FakeTestTypes) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(testtypesResource, c.ns, name), &testgroup.TestType{})

	return err
}

func (c *FakeTestTypes) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(testtypesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &testgroup.TestTypeList{})
	return err
}

func (c *FakeTestTypes) Get(name string) (result *testgroup.TestType, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(testtypesResource, c.ns, name), &testgroup.TestType{})

	if obj == nil {
		return nil, err
	}
	return obj.(*testgroup.TestType), err
}

func (c *FakeTestTypes) List(opts api.ListOptions) (result *testgroup.TestTypeList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(testtypesResource, c.ns, opts), &testgroup.TestTypeList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &testgroup.TestTypeList{}
	for _, item := range obj.(*testgroup.TestTypeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested testTypes.
func (c *FakeTestTypes) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(testtypesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched testType.
func (c *FakeTestTypes) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *testgroup.TestType, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(testtypesResource, c.ns, name, data, subresources...), &testgroup.TestType{})

	if obj == nil {
		return nil, err
	}
	return obj.(*testgroup.TestType), err
}
