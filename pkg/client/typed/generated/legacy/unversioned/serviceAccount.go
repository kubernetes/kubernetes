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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ServiceAccountNamespacer has methods to work with ServiceAccount resources in a namespace
type ServiceAccountNamespacer interface {
	ServiceAccounts(namespace string) ServiceAccountInterface
}

// ServiceAccountInterface has methods to work with ServiceAccount resources.
type ServiceAccountInterface interface {
	Create(*api.ServiceAccount) (*api.ServiceAccount, error)
	Update(*api.ServiceAccount) (*api.ServiceAccount, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.ServiceAccount, error)
	List(opts unversioned.ListOptions) (*api.ServiceAccountList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// serviceAccounts implements ServiceAccountInterface
type serviceAccounts struct {
	client *LegacyClient
	ns     string
}

// newServiceAccounts returns a ServiceAccounts
func newServiceAccounts(c *LegacyClient, namespace string) *serviceAccounts {
	return &serviceAccounts{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a serviceAccount and creates it.  Returns the server's representation of the serviceAccount, and an error, if there is any.
func (c *serviceAccounts) Create(serviceAccount *api.ServiceAccount) (result *api.ServiceAccount, err error) {
	result = &api.ServiceAccount{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("serviceAccounts").
		Body(serviceAccount).
		Do().
		Into(result)
	return
}

// Update takes the representation of a serviceAccount and updates it. Returns the server's representation of the serviceAccount, and an error, if there is any.
func (c *serviceAccounts) Update(serviceAccount *api.ServiceAccount) (result *api.ServiceAccount, err error) {
	result = &api.ServiceAccount{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("serviceAccounts").
		Name(serviceAccount.Name).
		Body(serviceAccount).
		Do().
		Into(result)
	return
}

// Delete takes name of the serviceAccount and deletes it. Returns an error if one occurs.
func (c *serviceAccounts) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("serviceAccounts").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("serviceAccounts").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the serviceAccount, and returns the corresponding serviceAccount object, and an error if there is any.
func (c *serviceAccounts) Get(name string) (result *api.ServiceAccount, err error) {
	result = &api.ServiceAccount{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("serviceAccounts").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ServiceAccounts that match those selectors.
func (c *serviceAccounts) List(opts unversioned.ListOptions) (result *api.ServiceAccountList, err error) {
	result = &api.ServiceAccountList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("serviceAccounts").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested serviceAccounts.
func (c *serviceAccounts) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("serviceAccounts").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
