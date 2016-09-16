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

package unversioned

import (
	api "k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	watch "k8s.io/kubernetes/pkg/watch"
)

// PodSecurityPoliciesGetter has a method to return a PodSecurityPolicyInterface.
// A group's client should implement this interface.
type PodSecurityPoliciesGetter interface {
	PodSecurityPolicies() PodSecurityPolicyInterface
}

// PodSecurityPolicyInterface has methods to work with PodSecurityPolicy resources.
type PodSecurityPolicyInterface interface {
	Create(*extensions.PodSecurityPolicy) (*extensions.PodSecurityPolicy, error)
	Update(*extensions.PodSecurityPolicy) (*extensions.PodSecurityPolicy, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*extensions.PodSecurityPolicy, error)
	List(opts api.ListOptions) (*extensions.PodSecurityPolicyList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.PodSecurityPolicy, err error)
	PodSecurityPolicyExpansion
}

// podSecurityPolicies implements PodSecurityPolicyInterface
type podSecurityPolicies struct {
	client *ExtensionsClient
}

// newPodSecurityPolicies returns a PodSecurityPolicies
func newPodSecurityPolicies(c *ExtensionsClient) *podSecurityPolicies {
	return &podSecurityPolicies{
		client: c,
	}
}

// Create takes the representation of a podSecurityPolicy and creates it.  Returns the server's representation of the podSecurityPolicy, and an error, if there is any.
func (c *podSecurityPolicies) Create(podSecurityPolicy *extensions.PodSecurityPolicy) (result *extensions.PodSecurityPolicy, err error) {
	result = &extensions.PodSecurityPolicy{}
	err = c.client.Post().
		Resource("podsecuritypolicies").
		Body(podSecurityPolicy).
		Do().
		Into(result)
	return
}

// Update takes the representation of a podSecurityPolicy and updates it. Returns the server's representation of the podSecurityPolicy, and an error, if there is any.
func (c *podSecurityPolicies) Update(podSecurityPolicy *extensions.PodSecurityPolicy) (result *extensions.PodSecurityPolicy, err error) {
	result = &extensions.PodSecurityPolicy{}
	err = c.client.Put().
		Resource("podsecuritypolicies").
		Name(podSecurityPolicy.Name).
		Body(podSecurityPolicy).
		Do().
		Into(result)
	return
}

// Delete takes name of the podSecurityPolicy and deletes it. Returns an error if one occurs.
func (c *podSecurityPolicies) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("podsecuritypolicies").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *podSecurityPolicies) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("podsecuritypolicies").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the podSecurityPolicy, and returns the corresponding podSecurityPolicy object, and an error if there is any.
func (c *podSecurityPolicies) Get(name string) (result *extensions.PodSecurityPolicy, err error) {
	result = &extensions.PodSecurityPolicy{}
	err = c.client.Get().
		Resource("podsecuritypolicies").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PodSecurityPolicies that match those selectors.
func (c *podSecurityPolicies) List(opts api.ListOptions) (result *extensions.PodSecurityPolicyList, err error) {
	result = &extensions.PodSecurityPolicyList{}
	err = c.client.Get().
		Resource("podsecuritypolicies").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested podSecurityPolicies.
func (c *podSecurityPolicies) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("podsecuritypolicies").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched podSecurityPolicy.
func (c *podSecurityPolicies) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.PodSecurityPolicy, err error) {
	result = &extensions.PodSecurityPolicy{}
	err = c.client.Patch(pt).
		Resource("podsecuritypolicies").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
