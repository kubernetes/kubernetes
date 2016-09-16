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

// SecretsGetter has a method to return a SecretInterface.
// A group's client should implement this interface.
type SecretsGetter interface {
	Secrets(namespace string) SecretInterface
}

// SecretInterface has methods to work with Secret resources.
type SecretInterface interface {
	Create(*v1.Secret) (*v1.Secret, error)
	Update(*v1.Secret) (*v1.Secret, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1.Secret, error)
	List(opts api.ListOptions) (*v1.SecretList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Secret, err error)
	SecretExpansion
}

// secrets implements SecretInterface
type secrets struct {
	client *CoreClient
	ns     string
}

// newSecrets returns a Secrets
func newSecrets(c *CoreClient, namespace string) *secrets {
	return &secrets{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a secret and creates it.  Returns the server's representation of the secret, and an error, if there is any.
func (c *secrets) Create(secret *v1.Secret) (result *v1.Secret, err error) {
	result = &v1.Secret{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("secrets").
		Body(secret).
		Do().
		Into(result)
	return
}

// Update takes the representation of a secret and updates it. Returns the server's representation of the secret, and an error, if there is any.
func (c *secrets) Update(secret *v1.Secret) (result *v1.Secret, err error) {
	result = &v1.Secret{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("secrets").
		Name(secret.Name).
		Body(secret).
		Do().
		Into(result)
	return
}

// Delete takes name of the secret and deletes it. Returns an error if one occurs.
func (c *secrets) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("secrets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *secrets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("secrets").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the secret, and returns the corresponding secret object, and an error if there is any.
func (c *secrets) Get(name string) (result *v1.Secret, err error) {
	result = &v1.Secret{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("secrets").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Secrets that match those selectors.
func (c *secrets) List(opts api.ListOptions) (result *v1.SecretList, err error) {
	result = &v1.SecretList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("secrets").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested secrets.
func (c *secrets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("secrets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched secret.
func (c *secrets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Secret, err error) {
	result = &v1.Secret{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("secrets").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
