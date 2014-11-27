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

// ReplicationControllersNamespacer has methods to work with ReplicationController resources in a namespace
type ReplicationControllersNamespacer interface {
	ReplicationControllers(namespace string) ReplicationControllerInterface
}

// ReplicationControllerInterface has methods to work with ReplicationController resources.
type ReplicationControllerInterface interface {
	List(selector labels.Selector) (*api.ReplicationControllerList, error)
	Get(name string) (*api.ReplicationController, error)
	Create(ctrl *api.ReplicationController) (*api.ReplicationController, error)
	Update(ctrl *api.ReplicationController) (*api.ReplicationController, error)
	Delete(name string) error
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// replicationControllers implements ReplicationControllersNamespacer interface
type replicationControllers struct {
	r  *Client
	ns string
}

// newReplicationControllers returns a PodsClient
func newReplicationControllers(c *Client, namespace string) *replicationControllers {
	return &replicationControllers{c, namespace}
}

// List takes a selector, and returns the list of replication controllers that match that selector.
func (c *replicationControllers) List(selector labels.Selector) (result *api.ReplicationControllerList, err error) {
	result = &api.ReplicationControllerList{}
	err = c.r.Get().Namespace(c.ns).Path("replicationControllers").SelectorParam("labels", selector).Do().Into(result)
	return
}

// Get returns information about a particular replication controller.
func (c *replicationControllers) Get(name string) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.r.Get().Namespace(c.ns).Path("replicationControllers").Path(name).Do().Into(result)
	return
}

// Create creates a new replication controller.
func (c *replicationControllers) Create(controller *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.r.Post().Namespace(c.ns).Path("replicationControllers").Body(controller).Do().Into(result)
	return
}

// Update updates an existing replication controller.
func (c *replicationControllers) Update(controller *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	if len(controller.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", controller)
		return
	}
	err = c.r.Put().Namespace(c.ns).Path("replicationControllers").Path(controller.Name).Body(controller).Do().Into(result)
	return
}

// Delete deletes an existing replication controller.
func (c *replicationControllers) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Path("replicationControllers").Path(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested controllers.
func (c *replicationControllers) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Namespace(c.ns).
		Path("watch").
		Path("replicationControllers").
		Param("resourceVersion", resourceVersion).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Watch()
}
