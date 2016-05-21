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
	"k8s.io/kubernetes/pkg/watch"
)

// NetworkPolicyNamespacer has methods to work with NetworkPolicy resources in a namespace
type NetworkPolicyNamespacer interface {
	NetworkPolicies(namespace string) NetworkPolicyInterface
}

// NetworkPolicyInterface exposes methods to work on NetworkPolicy resources.
type NetworkPolicyInterface interface {
	List(opts api.ListOptions) (*extensions.NetworkPolicyList, error)
	Get(name string) (*extensions.NetworkPolicy, error)
	Create(networkPolicy *extensions.NetworkPolicy) (*extensions.NetworkPolicy, error)
	Update(networkPolicy *extensions.NetworkPolicy) (*extensions.NetworkPolicy, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// NetworkPolicies implements NetworkPolicyNamespacer interface
type NetworkPolicies struct {
	r  *ExtensionsClient
	ns string
}

// newNetworkPolicies returns a NetworkPolicies
func newNetworkPolicies(c *ExtensionsClient, namespace string) *NetworkPolicies {
	return &NetworkPolicies{c, namespace}
}

// List returns a list of networkPolicy that match the label and field selectors.
func (c *NetworkPolicies) List(opts api.ListOptions) (result *extensions.NetworkPolicyList, err error) {
	result = &extensions.NetworkPolicyList{}
	err = c.r.Get().Namespace(c.ns).Resource("networkpolicies").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular networkPolicy.
func (c *NetworkPolicies) Get(name string) (result *extensions.NetworkPolicy, err error) {
	result = &extensions.NetworkPolicy{}
	err = c.r.Get().Namespace(c.ns).Resource("networkpolicies").Name(name).Do().Into(result)
	return
}

// Create creates a new networkPolicy.
func (c *NetworkPolicies) Create(networkPolicy *extensions.NetworkPolicy) (result *extensions.NetworkPolicy, err error) {
	result = &extensions.NetworkPolicy{}
	err = c.r.Post().Namespace(c.ns).Resource("networkpolicies").Body(networkPolicy).Do().Into(result)
	return
}

// Update updates an existing networkPolicy.
func (c *NetworkPolicies) Update(networkPolicy *extensions.NetworkPolicy) (result *extensions.NetworkPolicy, err error) {
	result = &extensions.NetworkPolicy{}
	err = c.r.Put().Namespace(c.ns).Resource("networkpolicies").Name(networkPolicy.Name).Body(networkPolicy).Do().Into(result)
	return
}

// Delete deletes a networkPolicy, returns error if one occurs.
func (c *NetworkPolicies) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("networkpolicies").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested networkPolicy.
func (c *NetworkPolicies) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("networkpolicies").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
