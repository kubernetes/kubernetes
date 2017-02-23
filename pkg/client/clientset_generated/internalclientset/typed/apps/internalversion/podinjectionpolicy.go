/*
Copyright 2017 The Kubernetes Authors.

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

package internalversion

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	api "k8s.io/kubernetes/pkg/api"
	apps "k8s.io/kubernetes/pkg/apis/apps"
)

// PodInjectionPoliciesGetter has a method to return a PodInjectionPolicyInterface.
// A group's client should implement this interface.
type PodInjectionPoliciesGetter interface {
	PodInjectionPolicies(namespace string) PodInjectionPolicyInterface
}

// PodInjectionPolicyInterface has methods to work with PodInjectionPolicy resources.
type PodInjectionPolicyInterface interface {
	Create(*apps.PodInjectionPolicy) (*apps.PodInjectionPolicy, error)
	Update(*apps.PodInjectionPolicy) (*apps.PodInjectionPolicy, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*apps.PodInjectionPolicy, error)
	List(opts v1.ListOptions) (*apps.PodInjectionPolicyList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *apps.PodInjectionPolicy, err error)
	PodInjectionPolicyExpansion
}

// podInjectionPolicies implements PodInjectionPolicyInterface
type podInjectionPolicies struct {
	client rest.Interface
	ns     string
}

// newPodInjectionPolicies returns a PodInjectionPolicies
func newPodInjectionPolicies(c *AppsClient, namespace string) *podInjectionPolicies {
	return &podInjectionPolicies{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a podInjectionPolicy and creates it.  Returns the server's representation of the podInjectionPolicy, and an error, if there is any.
func (c *podInjectionPolicies) Create(podInjectionPolicy *apps.PodInjectionPolicy) (result *apps.PodInjectionPolicy, err error) {
	result = &apps.PodInjectionPolicy{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		Body(podInjectionPolicy).
		Do().
		Into(result)
	return
}

// Update takes the representation of a podInjectionPolicy and updates it. Returns the server's representation of the podInjectionPolicy, and an error, if there is any.
func (c *podInjectionPolicies) Update(podInjectionPolicy *apps.PodInjectionPolicy) (result *apps.PodInjectionPolicy, err error) {
	result = &apps.PodInjectionPolicy{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		Name(podInjectionPolicy.Name).
		Body(podInjectionPolicy).
		Do().
		Into(result)
	return
}

// Delete takes name of the podInjectionPolicy and deletes it. Returns an error if one occurs.
func (c *podInjectionPolicies) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *podInjectionPolicies) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the podInjectionPolicy, and returns the corresponding podInjectionPolicy object, and an error if there is any.
func (c *podInjectionPolicies) Get(name string, options v1.GetOptions) (result *apps.PodInjectionPolicy, err error) {
	result = &apps.PodInjectionPolicy{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		Name(name).
		VersionedParams(&options, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PodInjectionPolicies that match those selectors.
func (c *podInjectionPolicies) List(opts v1.ListOptions) (result *apps.PodInjectionPolicyList, err error) {
	result = &apps.PodInjectionPolicyList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested podInjectionPolicies.
func (c *podInjectionPolicies) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched podInjectionPolicy.
func (c *podInjectionPolicies) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *apps.PodInjectionPolicy, err error) {
	result = &apps.PodInjectionPolicy{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("podinjectionpolicies").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
