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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/watch"
)

// PodDisruptionBudgetNamespacer has methods to work with PodDisruptionBudget resources in a namespace
type PodDisruptionBudgetNamespacer interface {
	PodDisruptionBudgets(namespace string) PodDisruptionBudgetInterface
}

// PodDisruptionBudgetInterface exposes methods to work on PodDisruptionBudget resources.
type PodDisruptionBudgetInterface interface {
	List(opts api.ListOptions) (*policy.PodDisruptionBudgetList, error)
	Get(name string) (*policy.PodDisruptionBudget, error)
	Create(podDisruptionBudget *policy.PodDisruptionBudget) (*policy.PodDisruptionBudget, error)
	Update(podDisruptionBudget *policy.PodDisruptionBudget) (*policy.PodDisruptionBudget, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
	UpdateStatus(podDisruptionBudget *policy.PodDisruptionBudget) (*policy.PodDisruptionBudget, error)
}

// podDisruptionBudget implements PodDisruptionBudgetNamespacer interface
type podDisruptionBudget struct {
	r  *PolicyClient
	ns string
}

// newPodDisruptionBudget returns a podDisruptionBudget
func newPodDisruptionBudget(c *PolicyClient, namespace string) *podDisruptionBudget {
	return &podDisruptionBudget{c, namespace}
}

// List returns a list of podDisruptionBudget that match the label and field selectors.
func (c *podDisruptionBudget) List(opts api.ListOptions) (result *policy.PodDisruptionBudgetList, err error) {
	result = &policy.PodDisruptionBudgetList{}
	err = c.r.Get().Namespace(c.ns).Resource("poddisruptionbudgets").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular podDisruptionBudget.
func (c *podDisruptionBudget) Get(name string) (result *policy.PodDisruptionBudget, err error) {
	result = &policy.PodDisruptionBudget{}
	err = c.r.Get().Namespace(c.ns).Resource("poddisruptionbudgets").Name(name).Do().Into(result)
	return
}

// Create creates a new podDisruptionBudget.
func (c *podDisruptionBudget) Create(podDisruptionBudget *policy.PodDisruptionBudget) (result *policy.PodDisruptionBudget, err error) {
	result = &policy.PodDisruptionBudget{}
	err = c.r.Post().Namespace(c.ns).Resource("poddisruptionbudgets").Body(podDisruptionBudget).Do().Into(result)
	return
}

// Update updates an existing podDisruptionBudget.
func (c *podDisruptionBudget) Update(podDisruptionBudget *policy.PodDisruptionBudget) (result *policy.PodDisruptionBudget, err error) {
	result = &policy.PodDisruptionBudget{}
	err = c.r.Put().Namespace(c.ns).Resource("poddisruptionbudgets").Name(podDisruptionBudget.Name).Body(podDisruptionBudget).Do().Into(result)
	return
}

// Delete deletes a podDisruptionBudget, returns error if one occurs.
func (c *podDisruptionBudget) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("poddisruptionbudgets").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested podDisruptionBudget.
func (c *podDisruptionBudget) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("poddisruptionbudgets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// UpdateStatus takes the name of the podDisruptionBudget and the new status.  Returns the server's representation of the podDisruptionBudget, and an error, if it occurs.
func (c *podDisruptionBudget) UpdateStatus(podDisruptionBudget *policy.PodDisruptionBudget) (result *policy.PodDisruptionBudget, err error) {
	result = &policy.PodDisruptionBudget{}
	err = c.r.Put().Namespace(c.ns).Resource("poddisruptionbudgets").Name(podDisruptionBudget.Name).SubResource("status").Body(podDisruptionBudget).Do().Into(result)
	return
}
