/*
Copyright 2015 The Kubernetes Authors.

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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	kclientlib "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeStorageClasses implements StorageClassInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeStorageClasses struct {
	Fake *FakeExperimental
}

// Ensure statically that FakeStorageClasses implements StorageClassInterface.
var _ kclientlib.StorageClassInterface = &FakeStorageClasses{}

func (c *FakeStorageClasses) Get(name string) (*extensions.StorageClass, error) {
	obj, err := c.Fake.Invokes(NewGetAction("storageclasses", "", name), &extensions.StorageClass{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.StorageClass), err
}

func (c *FakeStorageClasses) List(opts api.ListOptions) (*extensions.StorageClassList, error) {
	obj, err := c.Fake.Invokes(NewListAction("storageclasses", "", opts), &extensions.StorageClassList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.StorageClassList), err
}

func (c *FakeStorageClasses) Create(np *extensions.StorageClass) (*extensions.StorageClass, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("storageclasses", "", np), &extensions.StorageClass{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.StorageClass), err
}

func (c *FakeStorageClasses) Update(np *extensions.StorageClass) (*extensions.StorageClass, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("storageclasses", "", np), &extensions.StorageClass{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.StorageClass), err
}

func (c *FakeStorageClasses) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("storageclasses", "", name), &extensions.StorageClass{})
	return err
}

func (c *FakeStorageClasses) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("storageclasses", "", opts))
}
