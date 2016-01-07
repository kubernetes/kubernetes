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
	ThirdPartyResources(namespace string) ThirdPartyResourceInterface
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
	r  *ExtensionsClient
	ns string
}

func newThirdPartyResources(c *ExtensionsClient, namespace string) *thirdPartyResources {
	return &thirdPartyResources{c, namespace}
}

// Ensure statically that thirdPartyResources implements ThirdPartyResourcesInterface.
var _ ThirdPartyResourceInterface = &thirdPartyResources{}

func (c *thirdPartyResources) List(opts api.ListOptions) (result *extensions.ThirdPartyResourceList, err error) {
	result = &extensions.ThirdPartyResourceList{}
	err = c.r.Get().Namespace(c.ns).Resource("thirdpartyresources").VersionedParams(&opts, api.Scheme).Do().Into(result)
	return
}

// Get returns information about a particular daemon set.
func (c *thirdPartyResources) Get(name string) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Get().Namespace(c.ns).Resource("thirdpartyresources").Name(name).Do().Into(result)
	return
}

// Create creates a new daemon set.
func (c *thirdPartyResources) Create(daemon *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Post().Namespace(c.ns).Resource("thirdpartyresources").Body(daemon).Do().Into(result)
	return
}

// Update updates an existing daemon set.
func (c *thirdPartyResources) Update(daemon *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Put().Namespace(c.ns).Resource("thirdpartyresources").Name(daemon.Name).Body(daemon).Do().Into(result)
	return
}

// UpdateStatus updates an existing daemon set status
func (c *thirdPartyResources) UpdateStatus(daemon *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.r.Put().Namespace(c.ns).Resource("thirdpartyresources").Name(daemon.Name).SubResource("status").Body(daemon).Do().Into(result)
	return
}

// Delete deletes an existing daemon set.
func (c *thirdPartyResources) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("thirdpartyresources").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested daemon sets.
func (c *thirdPartyResources) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("thirdpartyresources").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
