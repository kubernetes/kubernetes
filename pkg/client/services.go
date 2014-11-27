/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// ServicesNamespacer has methods to work with Service resources in a namespace
type ServicesNamespacer interface {
	Services(namespace string) ServiceInterface
}

// ServiceInterface has methods to work with Service resources.
type ServiceInterface interface {
	List(selector labels.Selector) (*api.ServiceList, error)
	Get(name string) (*api.Service, error)
	Create(srv *api.Service) (*api.Service, error)
	Update(srv *api.Service) (*api.Service, error)
	Delete(name string) error
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// services implements PodsNamespacer interface
type services struct {
	r  *Client
	ns string
}

// newServices returns a PodsClient
func newServices(c *Client, namespace string) *services {
	return &services{c, namespace}
}

// List takes a selector, and returns the list of services that match that selector
func (c *services) List(selector labels.Selector) (result *api.ServiceList, err error) {
	result = &api.ServiceList{}
	err = c.r.Get().Namespace(c.ns).Path("services").SelectorParam("labels", selector).Do().Into(result)
	return
}

// Get returns information about a particular service.
func (c *services) Get(name string) (result *api.Service, err error) {
	result = &api.Service{}
	err = c.r.Get().Namespace(c.ns).Path("services").Path(name).Do().Into(result)
	return
}

// Create creates a new service.
func (c *services) Create(svc *api.Service) (result *api.Service, err error) {
	result = &api.Service{}
	err = c.r.Post().Namespace(c.ns).Path("services").Body(svc).Do().Into(result)
	return
}

// Update updates an existing service.
func (c *services) Update(svc *api.Service) (result *api.Service, err error) {
	result = &api.Service{}
	if len(svc.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", svc)
		return
	}
	err = c.r.Put().Namespace(c.ns).Path("services").Path(svc.Name).Body(svc).Do().Into(result)
	return
}

// Delete deletes an existing service.
func (c *services) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Path("services").Path(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested services.
func (c *services) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Namespace(c.ns).
		Path("watch").
		Path("services").
		Param("resourceVersion", resourceVersion).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Watch()
}
