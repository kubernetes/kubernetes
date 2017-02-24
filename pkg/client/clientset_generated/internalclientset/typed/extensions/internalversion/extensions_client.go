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
	rest "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/scheme"
)

type ExtensionsInterface interface {
	RESTClient() rest.Interface
	DaemonSetsGetter
	DeploymentsGetter
	IngressesGetter
	NetworkPoliciesGetter
	PodSecurityPoliciesGetter
	ReplicaSetsGetter
	ScalesGetter
	ThirdPartyResourcesGetter
}

// ExtensionsClient is used to interact with features provided by the extensions group.
type ExtensionsClient struct {
	restClient rest.Interface
}

func (c *ExtensionsClient) DaemonSets(namespace string) DaemonSetInterface {
	return newDaemonSets(c, namespace)
}

func (c *ExtensionsClient) Deployments(namespace string) DeploymentInterface {
	return newDeployments(c, namespace)
}

func (c *ExtensionsClient) Ingresses(namespace string) IngressInterface {
	return newIngresses(c, namespace)
}

func (c *ExtensionsClient) NetworkPolicies(namespace string) NetworkPolicyInterface {
	return newNetworkPolicies(c, namespace)
}

func (c *ExtensionsClient) PodSecurityPolicies() PodSecurityPolicyInterface {
	return newPodSecurityPolicies(c)
}

func (c *ExtensionsClient) ReplicaSets(namespace string) ReplicaSetInterface {
	return newReplicaSets(c, namespace)
}

func (c *ExtensionsClient) Scales(namespace string) ScaleInterface {
	return newScales(c, namespace)
}

func (c *ExtensionsClient) ThirdPartyResources() ThirdPartyResourceInterface {
	return newThirdPartyResources(c)
}

// NewForConfig creates a new ExtensionsClient for the given config.
func NewForConfig(c *rest.Config) (*ExtensionsClient, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := rest.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &ExtensionsClient{client}, nil
}

// NewForConfigOrDie creates a new ExtensionsClient for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *rest.Config) *ExtensionsClient {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new ExtensionsClient for the given RESTClient.
func New(c rest.Interface) *ExtensionsClient {
	return &ExtensionsClient{c}
}

func setConfigDefaults(config *rest.Config) error {
	g, err := scheme.Registry.Group("extensions")
	if err != nil {
		return err
	}

	config.APIPath = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = rest.DefaultKubernetesUserAgent()
	}
	if config.GroupVersion == nil || config.GroupVersion.Group != g.GroupVersion.Group {
		gv := g.GroupVersion
		config.GroupVersion = &gv
	}
	config.NegotiatedSerializer = scheme.Codecs

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
func (c *ExtensionsClient) RESTClient() rest.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
