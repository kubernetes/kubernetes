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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// AutoScaler has methods to work with AutoScaler resources in a namespace.
type AutoScalersNamespacer interface {
	AutoScalers(namespace string) AutoScalerInterface
}

// AutoScalerInterface has methods to work with AutoScaler resources.
type AutoScalerInterface interface {
	List(label labels.Selector, field fields.Selector) (*api.AutoScalerList, error)
	Create(item *api.AutoScaler) (*api.AutoScaler, error)
	Get(name string) (*api.AutoScaler, error)
	Update(item *api.AutoScaler) (*api.AutoScaler, error)
	UpdateStatus(item *api.AutoScaler) (*api.AutoScaler, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// autoscalers implements AutoScalerInterface.
type autoscalers struct {
	r  *Client
	ns string
}

// newAutoScalers returns an autoscalers instance
func newAutoScalers(c *Client, namespace string) *autoscalers {
	return &autoscalers{
		r:  c,
		ns: namespace,
	}
}

// List takes label and field selectors and returns the list of autoscalers
// that match those selectors.
func (c *autoscalers) List(label labels.Selector, field fields.Selector) (*api.AutoScalerList, error) {
	result := &api.AutoScalerList{}
	err := c.r.Get().Namespace(c.ns).Resource("autoscalers").LabelsSelectorParam(label).FieldsSelectorParam(field).Do().Into(result)
	return result, err
}

// Create creates a new AutoScaler.
func (c *autoscalers) Create(item *api.AutoScaler) (*api.AutoScaler, error) {
	result := &api.AutoScaler{}
	err := c.r.Post().Namespace(c.ns).Resource("autoscalers").Body(item).Do().Into(result)
	return result, err
}

// Get takes the name of the autoscaler and returns the corresponding
// AutoScaler object and any error, if it occurs.
func (c *autoscalers) Get(name string) (*api.AutoScaler, error) {
	result := &api.AutoScaler{}
	err := c.r.Get().Namespace(c.ns).Resource("autoscalers").Name(name).Do().Into(result)
	return result, err
}

// Update takes the representation of an autoscaler to update. Returns the
// server's representation of the autoscaler and any error, if it occurs.
func (c *autoscalers) Update(item *api.AutoScaler) (*api.AutoScaler, error) {
	result := &api.AutoScaler{}
	err := c.r.Put().Namespace(c.ns).Resource("autoscalers").Name(item.Name).Body(item).Do().Into(result)
	return result, err
}

// UpdateStatus takes the autoscaler and the new status. Returns the
// server's representation of the autoscaler and any error, if it occurs.
func (c *autoscalers) UpdateStatus(item *api.AutoScaler) (*api.AutoScaler, error) {
	result := &api.AutoScaler{}
	err := c.r.Put().Namespace(c.ns).Resource("autoscalers").Name(item.Name).SubResource("status").Body(item).Do().Into(result)
	return result, err
}

// Delete takes the name of an autoscaler and returns an error if one occurs
func (c *autoscalers) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("autoscalers").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested autoscalers.
func (c *autoscalers) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("autoscalers").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
