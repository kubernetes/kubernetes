/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

// ThirdPartyResourceNamespacer has methods to work with ThirdPartyResource resources in a namespace
type ThirdPartyResourceNamespacer interface {
	ThirdPartyResources() ThirdPartyResourceInterface
}

type ThirdPartyResourceInterface interface {
	List(opts api.ListOptions) (*extensions.ThirdPartyResourceList, error)
	Get(name string) (*extensions.ThirdPartyResource, error)
	Create(ctrl *extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error)
	Update(ctrl *extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error)
	UpdateStatus(ctrl *extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error)
	Delete(name string) error
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// thirdPartyResources implements DaemonsSetsNamespacer interface
type thirdPartyResources struct {
	r *ExtensionsClient
}

func newThirdPartyResources(c *ExtensionsClient) *thirdPartyResources {
	return &thirdPartyResources{c}
}

// Ensure statically that thirdPartyResources implements ThirdPartyResourcesInterface.
var _ ThirdPartyResourceInterface = &thirdPartyResources{}

func (c *thirdPartyResources) List(opts api.ListOptions) (result *extensions.ThirdPartyResourceList, err error) {
	result = &extensions.ThirdPartyResourceList{}
	err = c.r.Get().Resource("thirdpartyresources").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular third party resource.
func (c *thirdPartyResources) Get(name string) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Get().Resource("thirdpartyresources").Name(name).Do().Into(result)
	return
}

// Create creates a new third party resource.
func (c *thirdPartyResources) Create(resource *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Post().Resource("thirdpartyresources").Body(resource).Do().Into(result)
	return
}

// Update updates an existing third party resource.
func (c *thirdPartyResources) Update(resource *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Put().Resource("thirdpartyresources").Name(resource.Name).Body(resource).Do().Into(result)
	return
}

// UpdateStatus updates an existing third party resource status
func (c *thirdPartyResources) UpdateStatus(resource *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Put().Resource("thirdpartyresources").Name(resource.Name).SubResource("status").Body(resource).Do().Into(result)
	return
}

// Delete deletes an existing third party resource.
func (c *thirdPartyResources) Delete(name string) error {
	return c.r.Delete().Resource("thirdpartyresources").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested third party resources.
func (c *thirdPartyResources) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Resource("thirdpartyresources").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
