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

// FakeThirdPartyResources implements ThirdPartyResourceInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeThirdPartyResources struct {
	Fake *FakeExperimental
}

// Ensure statically that FakeThirdPartyResources implements DaemonInterface.
var _ kclientlib.ThirdPartyResourceInterface = &FakeThirdPartyResources{}

func (c *FakeThirdPartyResources) Get(name string) (*extensions.ThirdPartyResource, error) {
	obj, err := c.Fake.Invokes(NewGetAction("thirdpartyresources", "", name), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}

func (c *FakeThirdPartyResources) List(opts api.ListOptions) (*extensions.ThirdPartyResourceList, error) {
	obj, err := c.Fake.Invokes(NewListAction("thirdpartyresources", "", opts), &extensions.ThirdPartyResourceList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResourceList), err
}

func (c *FakeThirdPartyResources) Create(daemon *extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("thirdpartyresources", "", daemon), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}

func (c *FakeThirdPartyResources) Update(daemon *extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("thirdpartyresources", "", daemon), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}

func (c *FakeThirdPartyResources) UpdateStatus(daemon *extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("thirdpartyresources", "status", "", daemon), &extensions.ThirdPartyResource{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.ThirdPartyResource), err
}

func (c *FakeThirdPartyResources) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("thirdpartyresources", "", name), &extensions.ThirdPartyResource{})
	return err
}

func (c *FakeThirdPartyResources) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("thirdpartyresources", "", opts))
}
