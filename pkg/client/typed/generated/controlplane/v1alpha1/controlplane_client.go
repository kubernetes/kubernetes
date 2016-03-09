/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package v1alpha1

import (
	api "k8s.io/kubernetes/pkg/api"
	registered "k8s.io/kubernetes/pkg/apimachinery/registered"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

type ControlplaneInterface interface {
	ClustersGetter
}

// ControlplaneClient is used to interact with features provided by the Controlplane group.
type ControlplaneClient struct {
	*restclient.RESTClient
}

func (c *ControlplaneClient) Clusters() ClusterInterface {
	return newClusters(c)
}

// NewForConfig creates a new ControlplaneClient for the given config.
func NewForConfig(c *restclient.Config) (*ControlplaneClient, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &ControlplaneClient{client}, nil
}

// NewForConfigOrDie creates a new ControlplaneClient for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *ControlplaneClient {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new ControlplaneClient for the given RESTClient.
func New(c *restclient.RESTClient) *ControlplaneClient {
	return &ControlplaneClient{c}
}

func setConfigDefaults(config *restclient.Config) error {
	// if controlplane group is not registered, return an error
	g, err := registered.Group("controlplane")
	if err != nil {
		return err
	}
	config.APIPath = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

	config.Codec = api.Codecs.LegacyCodec(*config.GroupVersion)
	if config.QPS == 0 {
		config.QPS = 5
	}
	if config.Burst == 0 {
		config.Burst = 10
	}
	return nil
}
