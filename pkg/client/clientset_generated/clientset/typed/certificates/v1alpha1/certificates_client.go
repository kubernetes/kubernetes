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

package v1alpha1

import (
	fmt "fmt"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	api "k8s.io/kubernetes/pkg/api"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

type CertificatesV1alpha1Interface interface {
	RESTClient() restclient.Interface
	CertificateSigningRequestsGetter
}

// CertificatesV1alpha1Client is used to interact with features provided by the certificates.k8s.io group.
type CertificatesV1alpha1Client struct {
	restClient restclient.Interface
}

func (c *CertificatesV1alpha1Client) CertificateSigningRequests() CertificateSigningRequestInterface {
	return newCertificateSigningRequests(c)
}

// NewForConfig creates a new CertificatesV1alpha1Client for the given config.
func NewForConfig(c *restclient.Config) (*CertificatesV1alpha1Client, error) {
	config := *c
	if err := setConfigDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}
	return &CertificatesV1alpha1Client{client}, nil
}

// NewForConfigOrDie creates a new CertificatesV1alpha1Client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *CertificatesV1alpha1Client {
	client, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return client
}

// New creates a new CertificatesV1alpha1Client for the given RESTClient.
func New(c restclient.Interface) *CertificatesV1alpha1Client {
	return &CertificatesV1alpha1Client{c}
}

func setConfigDefaults(config *restclient.Config) error {
	gv, err := schema.ParseGroupVersion("certificates.k8s.io/v1alpha1")
	if err != nil {
		return err
	}
	// if certificates.k8s.io/v1alpha1 is not enabled, return an error
	if !api.Registry.IsEnabledVersion(gv) {
		return fmt.Errorf("certificates.k8s.io/v1alpha1 is not enabled")
	}
	config.APIPath = "/apis"
	if config.UserAgent == "" {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	copyGroupVersion := gv
	config.GroupVersion = &copyGroupVersion

	config.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: api.Codecs}

	return nil
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *CertificatesV1alpha1Client) RESTClient() restclient.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
