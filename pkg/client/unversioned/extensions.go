/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/restclient"
)

// Interface holds the experimental methods for clients of Kubernetes
// to allow mock testing.
// Features of Extensions group are not supported and may be changed or removed in
// incompatible ways at any time.
type ExtensionsInterface interface {
	ScaleNamespacer
	DaemonSetsNamespacer
	DeploymentsNamespacer
	JobsNamespacer
	IngressNamespacer
	NetworkPolicyNamespacer
	ThirdPartyResourceNamespacer
	ReplicaSetsNamespacer
	PodSecurityPoliciesInterface
}

// ExtensionsClient is used to interact with experimental Kubernetes features.
// Features of Extensions group are not supported and may be changed or removed in
// incompatible ways at any time.
type ExtensionsClient struct {
	*restclient.RESTClient
}

func (c *ExtensionsClient) PodSecurityPolicies() PodSecurityPolicyInterface {
	return newPodSecurityPolicy(c)
}

func (c *ExtensionsClient) Scales(namespace string) ScaleInterface {
	return newScales(c, namespace)
}

func (c *ExtensionsClient) DaemonSets(namespace string) DaemonSetInterface {
	return newDaemonSets(c, namespace)
}

func (c *ExtensionsClient) Deployments(namespace string) DeploymentInterface {
	return newDeployments(c, namespace)
}

func (c *ExtensionsClient) Jobs(namespace string) JobInterface {
	return newJobs(c, namespace)
}

func (c *ExtensionsClient) Ingress(namespace string) IngressInterface {
	return newIngress(c, namespace)
}

func (c *ExtensionsClient) NetworkPolicies(namespace string) NetworkPolicyInterface {
	return newNetworkPolicies(c, namespace)
}

func (c *ExtensionsClient) ThirdPartyResources() ThirdPartyResourceInterface {
	return newThirdPartyResources(c)
}

func (c *ExtensionsClient) ReplicaSets(namespace string) ReplicaSetInterface {
	return newReplicaSets(c, namespace)
}

// NewExtensions creates a new ExtensionsClient for the given config. This client
// provides access to experimental Kubernetes features.
// Features of Extensions group are not supported and may be changed or removed in
// incompatible ways at any time.
func NewExtensions(c *restclient.Config) (*ExtensionsClient, error) {
	config := *c
	if err := setGroupDefaults(extensions.GroupName, &config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &ExtensionsClient{client}, nil
}

// NewExtensionsOrDie creates a new ExtensionsClient for the given config and
// panics if there is an error in the config.
// Features of Extensions group are not supported and may be changed or removed in
// incompatible ways at any time.
func NewExtensionsOrDie(c *restclient.Config) *ExtensionsClient {
	client, err := NewExtensions(c)
	if err != nil {
		panic(err)
	}
	return client
}
