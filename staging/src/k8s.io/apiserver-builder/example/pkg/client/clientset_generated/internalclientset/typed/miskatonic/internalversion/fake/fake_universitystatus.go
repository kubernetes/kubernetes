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

// FakeUniversityStatuses implements UniversityStatusInterface
type FakeUniversityStatuses struct {
	Fake *FakeMiskatonic
	ns   string
}

var universitystatusesResource = schema.GroupVersionResource{Group: "miskatonic", Version: "", Resource: "universitystatuses"}

func (c *FakeUniversityStatuses) Create(universityStatus *miskatonic.UniversityStatus) (result *miskatonic.UniversityStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(universitystatusesResource, c.ns, universityStatus), &miskatonic.UniversityStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversityStatus), err
}

func (c *FakeUniversityStatuses) Update(universityStatus *miskatonic.UniversityStatus) (result *miskatonic.UniversityStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(universitystatusesResource, c.ns, universityStatus), &miskatonic.UniversityStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversityStatus), err
}

func (c *FakeUniversityStatuses) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(universitystatusesResource, c.ns, name), &miskatonic.UniversityStatus{})

	return err
}

func (c *FakeUniversityStatuses) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(universitystatusesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &miskatonic.UniversityStatusList{})
	return err
}

func (c *FakeUniversityStatuses) Get(name string, options v1.GetOptions) (result *miskatonic.UniversityStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(universitystatusesResource, c.ns, name), &miskatonic.UniversityStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversityStatus), err
}

func (c *FakeUniversityStatuses) List(opts v1.ListOptions) (result *miskatonic.UniversityStatusList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(universitystatusesResource, c.ns, opts), &miskatonic.UniversityStatusList{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversityStatusList), err
}

// Watch returns a watch.Interface that watches the requested universityStatuses.
func (c *FakeUniversityStatuses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(universitystatusesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched universityStatus.
func (c *FakeUniversityStatuses) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *miskatonic.UniversityStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(universitystatusesResource, c.ns, name, data, subresources...), &miskatonic.UniversityStatus{})

	if obj == nil {
		return nil, err
	}
	return obj.(*miskatonic.UniversityStatus), err
}
