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

package app

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
)

var (
	groupVersions = []schema.GroupVersion{certificates.SchemeGroupVersion}
)

// GKESigner uses external calls to GKE in order to sign certificate signing
// requests.
type GKESigner struct {
	webhook        *webhook.GenericWebhook
	kubeConfigFile string
	retryBackoff   time.Duration
}

// NewGKESigner will create a new instance of a GKESigner.
func NewGKESigner(kubeConfigFile string, retryBackoff time.Duration) (*GKESigner, error) {
	webhook, err := webhook.NewGenericWebhook(api.Registry, api.Codecs, kubeConfigFile, groupVersions, retryBackoff)
	if err != nil {
		return nil, err
	}

	return &GKESigner{
		webhook:        webhook,
		kubeConfigFile: kubeConfigFile,
		retryBackoff:   retryBackoff,
	}, nil
}

// Sign will make an external call to GKE order to sign the given
// *certificates.CertificateSigningRequest, using the GKESigner's
// kubeConfigFile.
func (s *GKESigner) Sign(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	result := s.webhook.WithExponentialBackoff(func() rest.Result {
		return s.webhook.RestClient.Post().Body(csr).Do()
	})

	if err := result.Error(); err != nil {
		return nil, s.webhookError(csr, err)
	}

	var statusCode int
	if result.StatusCode(&statusCode); statusCode < 200 || statusCode >= 300 {
		return nil, s.webhookError(csr, fmt.Errorf("received unsuccessful response code from webhook: %d", statusCode))
	}

	result_csr := &certificates.CertificateSigningRequest{}

	if err := result.Into(result_csr); err != nil {
		return nil, s.webhookError(result_csr, err)
	}

	// Keep the original CSR intact, and only update fields we expect to change.
	csr.Status.Certificate = result_csr.Status.Certificate
	return csr, nil
}

func (s *GKESigner) webhookError(csr *certificates.CertificateSigningRequest, err error) error {
	glog.V(2).Infof("error contacting webhook backend: %s", err)
	return err
}
