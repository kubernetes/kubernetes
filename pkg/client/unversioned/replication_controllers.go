/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
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
	UpdateStatus(ctrl *api.ReplicationController) (*api.ReplicationController, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
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
	err = c.r.Get().Namespace(c.ns).Resource("replicationControllers").LabelsSelectorParam(selector).Do().Into(result)
	return
}

// Get returns information about a particular replication controller.
func (c *replicationControllers) Get(name string) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.r.Get().Namespace(c.ns).Resource("replicationControllers").Name(name).Do().Into(result)
	return
}

// Create creates a new replication controller.
func (c *replicationControllers) Create(controller *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.r.Post().Namespace(c.ns).Resource("replicationControllers").Body(controller).Do().Into(result)
	return
}

// Update updates an existing replication controller.
func (c *replicationControllers) Update(controller *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.r.Put().Namespace(c.ns).Resource("replicationControllers").Name(controller.Name).Body(controller).Do().Into(result)
	return
}

// UpdateStatus updates an existing replication controller status
func (c *replicationControllers) UpdateStatus(controller *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.r.Put().Namespace(c.ns).Resource("replicationControllers").Name(controller.Name).SubResource("status").Body(controller).Do().Into(result)
	return
}

// Delete deletes an existing replication controller.
func (c *replicationControllers) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("replicationControllers").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested controllers.
func (c *replicationControllers) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("replicationControllers").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
