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
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	innsmouth "k8s.io/apiserver-builder/example/pkg/apis/innsmouth"
	testing "k8s.io/client-go/testing"
)

// FakeDeepOneStatuses implements DeepOneStatusInterface
type FakeDeepOneStatuses struct {
	Fake *FakeInnsmouth
	ns   string
}

var deeponestatusesResource = schema.GroupVersionResource{Group: "innsmouth", Version: "", Resource: "deeponestatuses"}

func (c *FakeDeepOneStatuses) Create(deepOneStatus *innsmouth.DeepOneStatus) (result *innsmouth.DeepOneStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(deeponestatusesResource, c.ns, deepOneStatus), &innsmouth.DeepOneStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneStatus), err
}

func (c *FakeDeepOneStatuses) Update(deepOneStatus *innsmouth.DeepOneStatus) (result *innsmouth.DeepOneStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(deeponestatusesResource, c.ns, deepOneStatus), &innsmouth.DeepOneStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneStatus), err
}

func (c *FakeDeepOneStatuses) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(deeponestatusesResource, c.ns, name), &innsmouth.DeepOneStatus{})

	return err
}

func (c *FakeDeepOneStatuses) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(deeponestatusesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &innsmouth.DeepOneStatusList{})
	return err
}

func (c *FakeDeepOneStatuses) Get(name string, options v1.GetOptions) (result *innsmouth.DeepOneStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(deeponestatusesResource, c.ns, name), &innsmouth.DeepOneStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneStatus), err
}

func (c *FakeDeepOneStatuses) List(opts v1.ListOptions) (result *innsmouth.DeepOneStatusList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(deeponestatusesResource, c.ns, opts), &innsmouth.DeepOneStatusList{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneStatusList), err
}

// Watch returns a watch.Interface that watches the requested deepOneStatuses.
func (c *FakeDeepOneStatuses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(deeponestatusesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched deepOneStatus.
func (c *FakeDeepOneStatuses) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOneStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(deeponestatusesResource, c.ns, name, data, subresources...), &innsmouth.DeepOneStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneStatus), err
}
