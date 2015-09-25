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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// HorizontalPodAutoscalersNamespacer has methods to work with HorizontalPodAutoscaler resources in a namespace
type HorizontalPodAutoscalersNamespacer interface {
	HorizontalPodAutoscalers(namespace string) HorizontalPodAutoscalerInterface
}

// HorizontalPodAutoscalerInterface has methods to work with HorizontalPodAutoscaler resources.
type HorizontalPodAutoscalerInterface interface {
	List(label labels.Selector, field fields.Selector) (*extensions.HorizontalPodAutoscalerList, error)
	Get(name string) (*extensions.HorizontalPodAutoscaler, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error)
	Update(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error)
	UpdateStatus(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// horizontalPodAutoscalers implements HorizontalPodAutoscalersNamespacer interface
type horizontalPodAutoscalers struct {
	client *ExtensionsClient
	ns     string
}

// newHorizontalPodAutoscalers returns a horizontalPodAutoscalers
func newHorizontalPodAutoscalers(c *ExtensionsClient, namespace string) *horizontalPodAutoscalers {
	return &horizontalPodAutoscalers{
		client: c,
		ns:     namespace,
	}
}

// List takes label and field selectors, and returns the list of horizontalPodAutoscalers that match those selectors.
func (c *horizontalPodAutoscalers) List(label labels.Selector, field fields.Selector) (result *extensions.HorizontalPodAutoscalerList, err error) {
	result = &extensions.HorizontalPodAutoscalerList{}
	err = c.client.Get().Namespace(c.ns).Resource("horizontalPodAutoscalers").LabelsSelectorParam(label).FieldsSelectorParam(field).Do().Into(result)
	return
}

// Get takes the name of the horizontalPodAutoscaler, and returns the corresponding HorizontalPodAutoscaler object, and an error if it occurs
func (c *horizontalPodAutoscalers) Get(name string) (result *extensions.HorizontalPodAutoscaler, err error) {
	result = &extensions.HorizontalPodAutoscaler{}
	err = c.client.Get().Namespace(c.ns).Resource("horizontalPodAutoscalers").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the horizontalPodAutoscaler and deletes it.  Returns an error if one occurs.
func (c *horizontalPodAutoscalers) Delete(name string, options *api.DeleteOptions) error {
	// TODO: to make this reusable in other client libraries
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("horizontalPodAutoscalers").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion())
	if err != nil {
		return err
	}
	return c.client.Delete().Namespace(c.ns).Resource("horizontalPodAutoscalers").Name(name).Body(body).Do().Error()
}

// Create takes the representation of a horizontalPodAutoscaler and creates it.  Returns the server's representation of the horizontalPodAutoscaler, and an error, if it occurs.
func (c *horizontalPodAutoscalers) Create(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (result *extensions.HorizontalPodAutoscaler, err error) {
	result = &extensions.HorizontalPodAutoscaler{}
	err = c.client.Post().Namespace(c.ns).Resource("horizontalPodAutoscalers").Body(horizontalPodAutoscaler).Do().Into(result)
	return
}

// Update takes the representation of a horizontalPodAutoscaler and updates it.  Returns the server's representation of the horizontalPodAutoscaler, and an error, if it occurs.
func (c *horizontalPodAutoscalers) Update(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (result *extensions.HorizontalPodAutoscaler, err error) {
	result = &extensions.HorizontalPodAutoscaler{}
	err = c.client.Put().Namespace(c.ns).Resource("horizontalPodAutoscalers").Name(horizontalPodAutoscaler.Name).Body(horizontalPodAutoscaler).Do().Into(result)
	return
}

// UpdateStatus takes the representation of a horizontalPodAutoscaler and updates it.  Returns the server's representation of the horizontalPodAutoscaler, and an error, if it occurs.
func (c *horizontalPodAutoscalers) UpdateStatus(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (result *extensions.HorizontalPodAutoscaler, err error) {
	result = &extensions.HorizontalPodAutoscaler{}
	err = c.client.Put().Namespace(c.ns).Resource("horizontalPodAutoscalers").Name(horizontalPodAutoscaler.Name).SubResource("status").Body(horizontalPodAutoscaler).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested horizontalPodAutoscalers.
func (c *horizontalPodAutoscalers) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("horizontalPodAutoscalers").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
