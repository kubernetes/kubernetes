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

package concerto_cloud

import (
	"io"

	"github.com/scalingdata/gcfg"

	"k8s.io/kubernetes/pkg/cloudprovider"
)

// ProviderName is the name under which Concerto cloud provider is registered
const ProviderName = "concerto"

// ConcertoCloud is an implementation of Interface, LoadBalancer, and Instances
// for Flexiant Concerto.
type ConcertoCloud struct {
	// Abstracting access to Concerto API
	service ConcertoAPIService
}

// ConcertoConfig holds the Concerto cloud provider configuration.
// Example config file contents :
//
//	# concerto-cloud.conf
//	[connection]
//	apiendpoint = https://localhost:8443/
//	cert = /etc/concerto/api/cert.pem
//	key = /etc/concerto/api/private/key.pem
//
type ConcertoConfig struct {
	Connection struct {
		APIEndpoint string `gcfg:"apiendpoint"`
		CACert      string `gcfg:"cacert"`
		Cert        string `gcfg:"cert"`
		Key         string `gcfg:"key"`
	}
}

func init() {
	cloudprovider.RegisterCloudProvider(
		ProviderName,
		func(config io.Reader) (cloudprovider.Interface, error) {
			cc, err := newConcertoCloud(config)
			return cc, err
		})
}

// newConcertoCloud creates a new instance of ConcertoCloud.
func newConcertoCloud(config io.Reader) (*ConcertoCloud, error) {
	concertoConfig := ConcertoConfig{}

	err := gcfg.ReadInto(&concertoConfig, config)
	if err != nil {
		return nil, err
	}

	apiService, err := buildConcertoRESTClient(concertoConfig)
	if err != nil {
		return nil, err
	}
	return &ConcertoCloud{service: apiService}, nil
}
