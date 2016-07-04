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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/restclient"
)

// Interface holds the methods for clients of Kubernetes to allow mock testing.
type RbacInterface interface {
	RoleBindingsNamespacer
	RolesNamespacer
	ClusterRoleBindings
	ClusterRoles
}

type RbacClient struct {
	*restclient.RESTClient
}

func (c *RbacClient) RoleBindings(namespace string) RoleBindingInterface {
	return newRoleBindings(c, namespace)
}

func (c *RbacClient) Roles(namespace string) RoleInterface {
	return newRoles(c, namespace)
}

func (c *RbacClient) ClusterRoleBindings() ClusterRoleBindingInterface {
	return newClusterRoleBindings(c)
}

func (c *RbacClient) ClusterRoles() ClusterRoleInterface {
	return newClusterRoles(c)
}

// NewRbac creates a new RbacClient for the given config.
func NewRbac(c *restclient.Config) (*RbacClient, error) {
	config := *c
	if err := setRbacDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &RbacClient{client}, nil
}

// NewRbacOrDie creates a new RbacClient for the given config and
// panics if there is an error in the config.
func NewRbacOrDie(c *restclient.Config) *RbacClient {
	client, err := NewRbac(c)
	if err != nil {
		panic(err)
	}
	return client
}

func setRbacDefaults(config *restclient.Config) error {
	// if rbac group is not registered, return an error
	g, err := registered.Group(rbac.GroupName)
	if err != nil {
		return err
	}
	config.APIPath = defaultAPIPath
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}

	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

	config.Codec = api.Codecs.LegacyCodec(*config.GroupVersion)
	config.NegotiatedSerializer = api.Codecs
	return nil
}
