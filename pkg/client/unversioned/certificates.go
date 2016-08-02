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
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/client/restclient"
)

// Interface holds the methods for clients of Kubernetes to allow mock testing.
type CertificatesInterface interface {
	CertificateSigningRequests() CertificateSigningRequestInterface
}

type CertificatesClient struct {
	*restclient.RESTClient
}

func (c *CertificatesClient) CertificateSigningRequests() CertificateSigningRequestInterface {
	return newCertificateSigningRequests(c)
}

// NewCertificates creates a new CertificatesClient for the given config.
func NewCertificates(c *restclient.Config) (*CertificatesClient, error) {
	config := *c
	if err := setCertificatesDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &CertificatesClient{client}, nil
}

// NewCertificatesOrDie creates a new CertificatesClient for the given config and
// panics if there is an error in the config.
func NewCertificatesOrDie(c *restclient.Config) *CertificatesClient {
	client, err := NewCertificates(c)
	if err != nil {
		panic(err)
	}
	return client
}

func setCertificatesDefaults(config *restclient.Config) error {
	setGroupDefaults(certificates.GroupName, config)
	if config.QPS == 0 {
		config.QPS = 5
	}
	if config.Burst == 0 {
		config.Burst = 10
	}
	return nil
}
