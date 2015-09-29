/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type ServiceAccountsNamespacer interface {
	ServiceAccounts(namespace string) ServiceAccountsInterface
}

type ServiceAccountsInterface interface {
	Create(serviceAccount *api.ServiceAccount) (*api.ServiceAccount, error)
	Update(serviceAccount *api.ServiceAccount) (*api.ServiceAccount, error)
	Delete(name string) error
	List(label labels.Selector, field fields.Selector) (*api.ServiceAccountList, error)
	Get(name string) (*api.ServiceAccount, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// serviceAccounts implements ServiceAccounts interface
type serviceAccounts struct {
	client    *Client
	namespace string
}

// newServiceAccounts returns a new serviceAccounts object.
func newServiceAccounts(c *Client, ns string) ServiceAccountsInterface {
	return &serviceAccounts{
		client:    c,
		namespace: ns,
	}
}

func (s *serviceAccounts) Create(serviceAccount *api.ServiceAccount) (*api.ServiceAccount, error) {
	result := &api.ServiceAccount{}
	err := s.client.Post().
		Namespace(s.namespace).
		Resource("serviceAccounts").
		Body(serviceAccount).
		Do().
		Into(result)

	return result, err
}

// List returns a list of serviceAccounts matching the selectors.
func (s *serviceAccounts) List(label labels.Selector, field fields.Selector) (*api.ServiceAccountList, error) {
	result := &api.ServiceAccountList{}

	err := s.client.Get().
		Namespace(s.namespace).
		Resource("serviceAccounts").
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Do().
		Into(result)

	return result, err
}

// Get returns the given serviceAccount, or an error.
func (s *serviceAccounts) Get(name string) (*api.ServiceAccount, error) {
	result := &api.ServiceAccount{}
	err := s.client.Get().
		Namespace(s.namespace).
		Resource("serviceAccounts").
		Name(name).
		Do().
		Into(result)

	return result, err
}

// Watch starts watching for serviceAccounts matching the given selectors.
func (s *serviceAccounts) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.client.Get().
		Prefix("watch").
		Namespace(s.namespace).
		Resource("serviceAccounts").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}

func (s *serviceAccounts) Delete(name string) error {
	return s.client.Delete().
		Namespace(s.namespace).
		Resource("serviceAccounts").
		Name(name).
		Do().
		Error()
}

func (s *serviceAccounts) Update(serviceAccount *api.ServiceAccount) (result *api.ServiceAccount, err error) {
	result = &api.ServiceAccount{}
	err = s.client.Put().
		Namespace(s.namespace).
		Resource("serviceAccounts").
		Name(serviceAccount.Name).
		Body(serviceAccount).
		Do().
		Into(result)

	return
}
