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

package certificates

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

// A WebhookSigner is capable of calling an external webhook in order to sign
// certificate signing requests.
type WebhookSigner struct {
	webhook        *webhook.GenericWebhook
	kubeConfigFile string
	retryBackoff   time.Duration
}

// NewWebhookSigner will create a new instance of a WebhookSigner.
func NewWebhookSigner(kubeConfigFile string, retryBackoff time.Duration) (*WebhookSigner, error) {
	webhook, err := webhook.NewGenericWebhook(api.Registry, api.Codecs, kubeConfigFile, groupVersions, retryBackoff)
	if err != nil {
		return nil, err
	}

	return &WebhookSigner{
		webhook:        webhook,
		kubeConfigFile: kubeConfigFile,
		retryBackoff:   retryBackoff,
	}, nil
}

// Sign will make a call to an external webhook based on the WebhookSigner's
// kubeConfigFile in order to sign the given *certificates.CertificateSigningRequest.
func (s *WebhookSigner) Sign(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
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

	if err := result.Into(csr); err != nil {
		return nil, s.webhookError(result_csr, err)
	}

	return result_csr, nil
}

func (s *WebhookSigner) webhookError(csr *certificates.CertificateSigningRequest, err error) error {
	glog.V(2).Infof("error contacting webhook backend: %s", err)
	return err
}
