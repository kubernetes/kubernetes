/*
Copyright 2016 The Kubernetes Authors.

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

package v1alpha1

import (
	api "k8s.io/client-go/1.4/pkg/api"
	v1alpha1 "k8s.io/client-go/1.4/pkg/apis/policy/v1alpha1"
	watch "k8s.io/client-go/1.4/pkg/watch"
)

// PodDisruptionBudgetsGetter has a method to return a PodDisruptionBudgetInterface.
// A group's client should implement this interface.
type PodDisruptionBudgetsGetter interface {
	PodDisruptionBudgets(namespace string) PodDisruptionBudgetInterface
}

// PodDisruptionBudgetInterface has methods to work with PodDisruptionBudget resources.
type PodDisruptionBudgetInterface interface {
	Create(*v1alpha1.PodDisruptionBudget) (*v1alpha1.PodDisruptionBudget, error)
	Update(*v1alpha1.PodDisruptionBudget) (*v1alpha1.PodDisruptionBudget, error)
	UpdateStatus(*v1alpha1.PodDisruptionBudget) (*v1alpha1.PodDisruptionBudget, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1alpha1.PodDisruptionBudget, error)
	List(opts api.ListOptions) (*v1alpha1.PodDisruptionBudgetList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.PodDisruptionBudget, err error)
	PodDisruptionBudgetExpansion
}

// podDisruptionBudgets implements PodDisruptionBudgetInterface
type podDisruptionBudgets struct {
	client *PolicyClient
	ns     string
}

// newPodDisruptionBudgets returns a PodDisruptionBudgets
func newPodDisruptionBudgets(c *PolicyClient, namespace string) *podDisruptionBudgets {
	return &podDisruptionBudgets{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a podDisruptionBudget and creates it.  Returns the server's representation of the podDisruptionBudget, and an error, if there is any.
func (c *podDisruptionBudgets) Create(podDisruptionBudget *v1alpha1.PodDisruptionBudget) (result *v1alpha1.PodDisruptionBudget, err error) {
	result = &v1alpha1.PodDisruptionBudget{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		Body(podDisruptionBudget).
		Do().
		Into(result)
	return
}

// Update takes the representation of a podDisruptionBudget and updates it. Returns the server's representation of the podDisruptionBudget, and an error, if there is any.
func (c *podDisruptionBudgets) Update(podDisruptionBudget *v1alpha1.PodDisruptionBudget) (result *v1alpha1.PodDisruptionBudget, err error) {
	result = &v1alpha1.PodDisruptionBudget{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		Name(podDisruptionBudget.Name).
		Body(podDisruptionBudget).
		Do().
		Into(result)
	return
}

func (c *podDisruptionBudgets) UpdateStatus(podDisruptionBudget *v1alpha1.PodDisruptionBudget) (result *v1alpha1.PodDisruptionBudget, err error) {
	result = &v1alpha1.PodDisruptionBudget{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		Name(podDisruptionBudget.Name).
		SubResource("status").
		Body(podDisruptionBudget).
		Do().
		Into(result)
	return
}

// Delete takes name of the podDisruptionBudget and deletes it. Returns an error if one occurs.
func (c *podDisruptionBudgets) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *podDisruptionBudgets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the podDisruptionBudget, and returns the corresponding podDisruptionBudget object, and an error if there is any.
func (c *podDisruptionBudgets) Get(name string) (result *v1alpha1.PodDisruptionBudget, err error) {
	result = &v1alpha1.PodDisruptionBudget{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PodDisruptionBudgets that match those selectors.
func (c *podDisruptionBudgets) List(opts api.ListOptions) (result *v1alpha1.PodDisruptionBudgetList, err error) {
	result = &v1alpha1.PodDisruptionBudgetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested podDisruptionBudgets.
func (c *podDisruptionBudgets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched podDisruptionBudget.
func (c *podDisruptionBudgets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.PodDisruptionBudget, err error) {
	result = &v1alpha1.PodDisruptionBudget{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
