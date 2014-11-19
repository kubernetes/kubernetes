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

// PerNodeControllersNamespacer has methods to work with PerNodeController resources in a namespace
type PerNodeControllersNamespacer interface {
	PerNodeControllers(namespace string) PerNodeControllerInterface
}

// PerNodeControllerInterface has methods to work with PerNodeController resources.
type PerNodeControllerInterface interface {
	List(selector labels.Selector) (*api.PerNodeControllerList, error)
	Get(name string) (*api.PerNodeController, error)
	Create(ctrl *api.PerNodeController) (*api.PerNodeController, error)
	Update(ctrl *api.PerNodeController) (*api.PerNodeController, error)
	Delete(name string) error
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// perNodeControllers implements PerNodeControllersNamespacer interface
type perNodeControllers struct {
	r  *Client
	ns string
}

// newPerNodeControllers returns a PodsClient
func newPerNodeControllers(c *Client, namespace string) *perNodeControllers {
	return &perNodeControllers{c, namespace}
}

// List takes a selector, and returns the list of per-node controllers that match that selector.
func (c *perNodeControllers) List(selector labels.Selector) (result *api.PerNodeControllerList, err error) {
	result = &api.PerNodeControllerList{}
	err = c.r.Get().Namespace(c.ns).Path("perNodeControllers").SelectorParam("labels", selector).Do().Into(result)
	return
}

// Get returns information about a particular per-node controller.
func (c *perNodeControllers) Get(name string) (result *api.PerNodeController, err error) {
	result = &api.PerNodeController{}
	err = c.r.Get().Namespace(c.ns).Path("perNodeControllers").Path(name).Do().Into(result)
	return
}

// Create creates a new per-node controller.
func (c *perNodeControllers) Create(controller *api.PerNodeController) (result *api.PerNodeController, err error) {
	result = &api.PerNodeController{}
	err = c.r.Post().Namespace(c.ns).Path("perNodeControllers").Body(controller).Do().Into(result)
	return
}

// Update updates an existing per-node controller.
func (c *perNodeControllers) Update(controller *api.PerNodeController) (result *api.PerNodeController, err error) {
	result = &api.PerNodeController{}
	if len(controller.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", controller)
		return
	}
	err = c.r.Put().Namespace(c.ns).Path("perNodeControllers").Path(controller.Name).Body(controller).Do().Into(result)
	return
}

// Delete deletes an existing per-node controller.
func (c *perNodeControllers) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Path("perNodeControllers").Path(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested controllers.
func (c *perNodeControllers) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Namespace(c.ns).
		Path("watch").
		Path("perNodeControllers").
		Param("resourceVersion", resourceVersion).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Watch()
}
