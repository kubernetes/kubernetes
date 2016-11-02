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

package internalversion

import (
	api "k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// DeploymentsGetter has a method to return a DeploymentInterface.
// A group's client should implement this interface.
type DeploymentsGetter interface {
	Deployments(namespace string) DeploymentInterface
}

// DeploymentInterface has methods to work with Deployment resources.
type DeploymentInterface interface {
	Create(*extensions.Deployment) (*extensions.Deployment, error)
	Update(*extensions.Deployment) (*extensions.Deployment, error)
	UpdateStatus(*extensions.Deployment) (*extensions.Deployment, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*extensions.Deployment, error)
	List(opts api.ListOptions) (*extensions.DeploymentList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.Deployment, err error)
	DeploymentExpansion
}

// deployments implements DeploymentInterface
type deployments struct {
	client restclient.Interface
	ns     string
}

// newDeployments returns a Deployments
func newDeployments(c *ExtensionsClient, namespace string) *deployments {
	return &deployments{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a deployment and creates it.  Returns the server's representation of the deployment, and an error, if there is any.
func (c *deployments) Create(deployment *extensions.Deployment) (result *extensions.Deployment, err error) {
	result = &extensions.Deployment{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("deployments").
		Body(deployment).
		Do().
		Into(result)
	return
}

// Update takes the representation of a deployment and updates it. Returns the server's representation of the deployment, and an error, if there is any.
func (c *deployments) Update(deployment *extensions.Deployment) (result *extensions.Deployment, err error) {
	result = &extensions.Deployment{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("deployments").
		Name(deployment.Name).
		Body(deployment).
		Do().
		Into(result)
	return
}

func (c *deployments) UpdateStatus(deployment *extensions.Deployment) (result *extensions.Deployment, err error) {
	result = &extensions.Deployment{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("deployments").
		Name(deployment.Name).
		SubResource("status").
		Body(deployment).
		Do().
		Into(result)
	return
}

// Delete takes name of the deployment and deletes it. Returns an error if one occurs.
func (c *deployments) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deployments").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *deployments) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deployments").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the deployment, and returns the corresponding deployment object, and an error if there is any.
func (c *deployments) Get(name string) (result *extensions.Deployment, err error) {
	result = &extensions.Deployment{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deployments").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Deployments that match those selectors.
func (c *deployments) List(opts api.ListOptions) (result *extensions.DeploymentList, err error) {
	result = &extensions.DeploymentList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deployments").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested deployments.
func (c *deployments) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("deployments").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched deployment.
func (c *deployments) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.Deployment, err error) {
	result = &extensions.Deployment{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("deployments").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
