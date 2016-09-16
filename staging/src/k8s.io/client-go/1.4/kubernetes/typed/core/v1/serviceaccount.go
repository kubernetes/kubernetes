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

package v1

import (
	api "k8s.io/client-go/1.4/pkg/api"
	v1 "k8s.io/client-go/1.4/pkg/api/v1"
	watch "k8s.io/client-go/1.4/pkg/watch"
)

// ServiceAccountsGetter has a method to return a ServiceAccountInterface.
// A group's client should implement this interface.
type ServiceAccountsGetter interface {
	ServiceAccounts(namespace string) ServiceAccountInterface
}

// ServiceAccountInterface has methods to work with ServiceAccount resources.
type ServiceAccountInterface interface {
	Create(*v1.ServiceAccount) (*v1.ServiceAccount, error)
	Update(*v1.ServiceAccount) (*v1.ServiceAccount, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1.ServiceAccount, error)
	List(opts api.ListOptions) (*v1.ServiceAccountList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ServiceAccount, err error)
	ServiceAccountExpansion
}

// serviceAccounts implements ServiceAccountInterface
type serviceAccounts struct {
	client *CoreClient
	ns     string
}

// newServiceAccounts returns a ServiceAccounts
func newServiceAccounts(c *CoreClient, namespace string) *serviceAccounts {
	return &serviceAccounts{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a serviceAccount and creates it.  Returns the server's representation of the serviceAccount, and an error, if there is any.
func (c *serviceAccounts) Create(serviceAccount *v1.ServiceAccount) (result *v1.ServiceAccount, err error) {
	result = &v1.ServiceAccount{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("serviceaccounts").
		Body(serviceAccount).
		Do().
		Into(result)
	return
}

// Update takes the representation of a serviceAccount and updates it. Returns the server's representation of the serviceAccount, and an error, if there is any.
func (c *serviceAccounts) Update(serviceAccount *v1.ServiceAccount) (result *v1.ServiceAccount, err error) {
	result = &v1.ServiceAccount{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("serviceaccounts").
		Name(serviceAccount.Name).
		Body(serviceAccount).
		Do().
		Into(result)
	return
}

// Delete takes name of the serviceAccount and deletes it. Returns an error if one occurs.
func (c *serviceAccounts) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("serviceaccounts").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *serviceAccounts) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("serviceaccounts").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the serviceAccount, and returns the corresponding serviceAccount object, and an error if there is any.
func (c *serviceAccounts) Get(name string) (result *v1.ServiceAccount, err error) {
	result = &v1.ServiceAccount{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("serviceaccounts").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ServiceAccounts that match those selectors.
func (c *serviceAccounts) List(opts api.ListOptions) (result *v1.ServiceAccountList, err error) {
	result = &v1.ServiceAccountList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("serviceaccounts").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested serviceAccounts.
func (c *serviceAccounts) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("serviceaccounts").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched serviceAccount.
func (c *serviceAccounts) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ServiceAccount, err error) {
	result = &v1.ServiceAccount{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("serviceaccounts").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
