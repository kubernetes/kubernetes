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
	"fmt"
	latest "k8s.io/kubernetes/pkg/api/latest"
	unversioned "k8s.io/kubernetes/pkg/client/unversioned"
)

type TestgroupInterface interface {
	TestTypeNamespacer
}

// TestgroupClient is used to interact with features provided by the Testgroup group.
type TestgroupClient struct {
	*unversioned.RESTClient
}

func (c *TestgroupClient) TestTypes(namespace string) TestTypeInterface {
	return newTestTypes(c, namespace)
}

// NewForConfig creates a new TestgroupClient for the given config.
func NewForConfig(c *unversioned.Config) (*TestgroupClient, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := unversioned.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &TestgroupClient{client}, nil
}

// NewForConfigOrDie creates a new TestgroupClient for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *unversioned.Config) *TestgroupClient {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new TestgroupClient for the given RESTClient.
func New(c *unversioned.RESTClient) *TestgroupClient {
	return &TestgroupClient{c}
}

func setConfigDefaults(config *unversioned.Config) error {
	// if testgroup group is not registered, return an error
	g, err := latest.Group("testgroup")
	if err != nil {
		return err
	}
	config.Prefix = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = unversioned.DefaultKubernetesUserAgent()
	}
	// TODO: Unconditionally set the config.Version, until we fix the config.
	//if config.Version == "" {
	copyGroupVersion := g.GroupVersion
	config.GroupVersion = &copyGroupVersion
	//}

	versionInterfaces, err := g.InterfacesFor(*config.GroupVersion)
	if err != nil {
		return fmt.Errorf("Testgroup API version '%s' is not recognized (valid values: %s)",
			config.GroupVersion, g.GroupVersions)
	}
	config.Codec = versionInterfaces.Codec
	if config.QPS == 0 {
		config.QPS = 5
	}
	if config.Burst == 0 {
		config.Burst = 10
	}
	return nil
}
