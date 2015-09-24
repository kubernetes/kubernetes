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
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// DeploymentsNamespacer has methods to work with Deployment resources in a namespace
type DeploymentsNamespacer interface {
	Deployments(namespace string) DeploymentInterface
}

// DeploymentInterface has methods to work with Deployment resources.
type DeploymentInterface interface {
	List(label labels.Selector, field fields.Selector) (*experimental.DeploymentList, error)
	Get(name string) (*experimental.Deployment, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(Deployment *experimental.Deployment) (*experimental.Deployment, error)
	Update(Deployment *experimental.Deployment) (*experimental.Deployment, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// deployments implements DeploymentInterface
type deployments struct {
	client *ExperimentalClient
	ns     string
}

// newDeployments returns a Deployments
func newDeployments(c *ExperimentalClient, namespace string) *deployments {
	return &deployments{
		client: c,
		ns:     namespace,
	}
}

// List takes label and field selectors, and returns the list of Deployments that match those selectors.
func (c *deployments) List(label labels.Selector, field fields.Selector) (result *experimental.DeploymentList, err error) {
	result = &experimental.DeploymentList{}
	err = c.client.Get().Namespace(c.ns).Resource("deployments").LabelsSelectorParam(label).FieldsSelectorParam(field).Do().Into(result)
	return
}

// Get takes name of the deployment, and returns the corresponding deployment object, and an error if there is any.
func (c *deployments) Get(name string) (result *experimental.Deployment, err error) {
	result = &experimental.Deployment{}
	err = c.client.Get().Namespace(c.ns).Resource("deployments").Name(name).Do().Into(result)
	return
}

// Delete takes name of the deployment and deletes it. Returns an error if one occurs.
func (c *deployments) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("deployments").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion())
	if err != nil {
		return err
	}
	return c.client.Delete().Namespace(c.ns).Resource("deployments").Name(name).Body(body).Do().Error()
}

// Create takes the representation of a deployment and creates it.  Returns the server's representation of the deployment, and an error, if there is any.
func (c *deployments) Create(deployment *experimental.Deployment) (result *experimental.Deployment, err error) {
	result = &experimental.Deployment{}
	err = c.client.Post().Namespace(c.ns).Resource("deployments").Body(deployment).Do().Into(result)
	return
}

// Update takes the representation of a deployment and updates it. Returns the server's representation of the deployment, and an error, if there is any.
func (c *deployments) Update(deployment *experimental.Deployment) (result *experimental.Deployment, err error) {
	result = &experimental.Deployment{}
	err = c.client.Put().Namespace(c.ns).Resource("deployments").Name(deployment.Name).Body(deployment).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested deployments.
func (c *deployments) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("deployments").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
