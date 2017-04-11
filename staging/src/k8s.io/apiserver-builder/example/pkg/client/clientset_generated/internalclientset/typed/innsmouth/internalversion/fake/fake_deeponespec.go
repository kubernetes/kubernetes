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

// FakeDeepOneSpecs implements DeepOneSpecInterface
type FakeDeepOneSpecs struct {
	Fake *FakeInnsmouth
	ns   string
}

var deeponespecsResource = schema.GroupVersionResource{Group: "innsmouth", Version: "", Resource: "deeponespecs"}

func (c *FakeDeepOneSpecs) Create(deepOneSpec *innsmouth.DeepOneSpec) (result *innsmouth.DeepOneSpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(deeponespecsResource, c.ns, deepOneSpec), &innsmouth.DeepOneSpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneSpec), err
}

func (c *FakeDeepOneSpecs) Update(deepOneSpec *innsmouth.DeepOneSpec) (result *innsmouth.DeepOneSpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(deeponespecsResource, c.ns, deepOneSpec), &innsmouth.DeepOneSpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneSpec), err
}

func (c *FakeDeepOneSpecs) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(deeponespecsResource, c.ns, name), &innsmouth.DeepOneSpec{})

	return err
}

func (c *FakeDeepOneSpecs) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(deeponespecsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &innsmouth.DeepOneSpecList{})
	return err
}

func (c *FakeDeepOneSpecs) Get(name string, options v1.GetOptions) (result *innsmouth.DeepOneSpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(deeponespecsResource, c.ns, name), &innsmouth.DeepOneSpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneSpec), err
}

func (c *FakeDeepOneSpecs) List(opts v1.ListOptions) (result *innsmouth.DeepOneSpecList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(deeponespecsResource, c.ns, opts), &innsmouth.DeepOneSpecList{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneSpecList), err
}

// Watch returns a watch.Interface that watches the requested deepOneSpecs.
func (c *FakeDeepOneSpecs) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(deeponespecsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched deepOneSpec.
func (c *FakeDeepOneSpecs) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOneSpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(deeponespecsResource, c.ns, name, data, subresources...), &innsmouth.DeepOneSpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOneSpec), err
}
