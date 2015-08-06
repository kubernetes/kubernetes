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

package client

import (
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	explatest "k8s.io/kubernetes/pkg/expapi/latest"
	"k8s.io/kubernetes/pkg/version"
)

// Interface holds the experimental methods for clients of Kubernetes
// to allow mock testing.
// Experimental features are not supported and may be changed or removed in
// incompatible ways at any time.
type ExperimentalInterface interface {
	VersionInterface
}

// ExperimentalClient is used to interact with experimental Kubernetes features.
// Experimental features are not supported and may be changed or removed in
// incompatible ways at any time.
type ExperimentalClient struct {
	*RESTClient
}

// ServerVersion retrieves and parses the server's version.
func (c *ExperimentalClient) ServerVersion() (*version.Info, error) {
	body, err := c.Get().AbsPath("/version").Do().Raw()
	if err != nil {
		return nil, err
	}
	var info version.Info
	err = json.Unmarshal(body, &info)
	if err != nil {
		return nil, fmt.Errorf("got '%s': %v", string(body), err)
	}
	return &info, nil
}

// ServerAPIVersions retrieves and parses the list of experimental API versions the
// server supports.
func (c *ExperimentalClient) ServerAPIVersions() (*api.APIVersions, error) {
	body, err := c.Get().UnversionedPath("").Do().Raw()
	if err != nil {
		return nil, err
	}
	var v api.APIVersions
	err = json.Unmarshal(body, &v)
	if err != nil {
		return nil, fmt.Errorf("got '%s': %v", string(body), err)
	}
	return &v, nil
}

// NewExperimental creates a new ExperimentalClient for the given config. This client
// provides access to experimental Kubernetes features.
// Experimental features are not supported and may be changed or removed in
// incompatible ways at any time.
func NewExperimental(c *Config) (*ExperimentalClient, error) {
	config := *c
	if err := setExperimentalDefaults(&config); err != nil {
		return nil, err
	}
	client, err := RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &ExperimentalClient{client}, nil
}

// NewExperimentalOrDie creates a new ExperimentalClient for the given config and
// panics if there is an error in the config.
// Experimental features are not supported and may be changed or removed in
// incompatible ways at any time.
func NewExperimentalOrDie(c *Config) *ExperimentalClient {
	client, err := NewExperimental(c)
	if err != nil {
		panic(err)
	}
	return client
}

func setExperimentalDefaults(config *Config) error {
	if config.Prefix == "" {
		config.Prefix = "/experimental"
	}
	if config.UserAgent == "" {
		config.UserAgent = DefaultKubernetesUserAgent()
	}
	if config.Version == "" {
		config.Version = explatest.Version
	}
	versionInterfaces, err := explatest.InterfacesFor(config.Version)
	if err != nil {
		return fmt.Errorf("Experimental API version '%s' is not recognized (valid values: %s)",
			config.Version, strings.Join(explatest.Versions, ", "))
	}
	if config.Codec == nil {
		config.Codec = versionInterfaces.Codec
	}
	if config.QPS == 0 {
		config.QPS = 5
	}
	if config.Burst == 0 {
		config.Burst = 10
	}
	return nil
}
