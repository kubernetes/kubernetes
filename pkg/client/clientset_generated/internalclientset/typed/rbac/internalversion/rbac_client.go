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
	api "k8s.io/kubernetes/pkg/api"
	registered "k8s.io/kubernetes/pkg/apimachinery/registered"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

type RbacInterface interface {
	RESTClient() restclient.Interface
	ClusterRolesGetter
	ClusterRoleBindingsGetter
	RolesGetter
	RoleBindingsGetter
}

// RbacClient is used to interact with features provided by the k8s.io/kubernetes/pkg/apimachinery/registered.Group group.
type RbacClient struct {
	restClient restclient.Interface
}

func (c *RbacClient) ClusterRoles() ClusterRoleInterface {
	return newClusterRoles(c)
}

func (c *RbacClient) ClusterRoleBindings() ClusterRoleBindingInterface {
	return newClusterRoleBindings(c)
}

func (c *RbacClient) Roles(namespace string) RoleInterface {
	return newRoles(c, namespace)
}

func (c *RbacClient) RoleBindings(namespace string) RoleBindingInterface {
	return newRoleBindings(c, namespace)
}

// NewForConfig creates a new RbacClient for the given config.
func NewForConfig(c *restclient.Config) (*RbacClient, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &RbacClient{client}, nil
}

// NewForConfigOrDie creates a new RbacClient for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *RbacClient {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new RbacClient for the given RESTClient.
func New(c restclient.Interface) *RbacClient {
	return &RbacClient{c}
}

func setConfigDefaults(config *restclient.Config) error {
	// if rbac group is not registered, return an error
	g, err := registered.Group("rbac.authorization.k8s.io")
	if err != nil {
		return err
	}
	config.APIPath = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != g.GroupVersion.Group {
		copyGroupVersion := g.GroupVersion
		config.GroupVersion = &copyGroupVersion
	}
	config.NegotiatedSerializer = api.Codecs

	if config.QPS == 0 {
		config.QPS = 5
	}
	if config.Burst == 0 {
		config.Burst = 10
	}
	return nil
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *RbacClient) RESTClient() restclient.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
