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
	miskatonic "k8s.io/apiserver-builder/example/pkg/apis/miskatonic"
	testing "k8s.io/client-go/testing"
)

// FakeUniversitySpecs implements UniversitySpecInterface
type FakeUniversitySpecs struct {
	Fake *FakeMiskatonic
	ns   string
}

var universityspecsResource = schema.GroupVersionResource{Group: "miskatonic", Version: "", Resource: "universityspecs"}

func (c *FakeUniversitySpecs) Create(universitySpec *miskatonic.UniversitySpec) (result *miskatonic.UniversitySpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(universityspecsResource, c.ns, universitySpec), &miskatonic.UniversitySpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversitySpec), err
}

func (c *FakeUniversitySpecs) Update(universitySpec *miskatonic.UniversitySpec) (result *miskatonic.UniversitySpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(universityspecsResource, c.ns, universitySpec), &miskatonic.UniversitySpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversitySpec), err
}

func (c *FakeUniversitySpecs) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(universityspecsResource, c.ns, name), &miskatonic.UniversitySpec{})

	return err
}

func (c *FakeUniversitySpecs) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(universityspecsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &miskatonic.UniversitySpecList{})
	return err
}

func (c *FakeUniversitySpecs) Get(name string, options v1.GetOptions) (result *miskatonic.UniversitySpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(universityspecsResource, c.ns, name), &miskatonic.UniversitySpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversitySpec), err
}

func (c *FakeUniversitySpecs) List(opts v1.ListOptions) (result *miskatonic.UniversitySpecList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(universityspecsResource, c.ns, opts), &miskatonic.UniversitySpecList{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversitySpecList), err
}

// Watch returns a watch.Interface that watches the requested universitySpecs.
func (c *FakeUniversitySpecs) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(universityspecsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched universitySpec.
func (c *FakeUniversitySpecs) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *miskatonic.UniversitySpec, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(universityspecsResource, c.ns, name, data, subresources...), &miskatonic.UniversitySpec{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversitySpec), err
}
