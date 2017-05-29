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
	"encoding/json"
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	capi "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/controller/certificates"
)

var (
	groupVersions = []schema.GroupVersion{capi.SchemeGroupVersion}
)

// GKESigner uses external calls to GKE in order to sign certificate signing
// requests.
type GKESigner struct {
	webhook        *webhook.GenericWebhook
	kubeConfigFile string
	retryBackoff   time.Duration
	recorder       record.EventRecorder
	client         clientset.Interface
}

// NewGKESigner will create a new instance of a GKESigner.
func NewGKESigner(kubeConfigFile string, retryBackoff time.Duration, recorder record.EventRecorder, client clientset.Interface) (*GKESigner, error) {
	webhook, err := webhook.NewGenericWebhook(api.Registry, api.Codecs, kubeConfigFile, groupVersions, retryBackoff)
	if err != nil {
		return nil, err
	}

	return &GKESigner{
		webhook:        webhook,
		kubeConfigFile: kubeConfigFile,
		retryBackoff:   retryBackoff,
		recorder:       recorder,
		client:         client,
	}, nil
}

func (s *GKESigner) handle(csr *capi.CertificateSigningRequest) error {
	if !certificates.IsCertificateRequestApproved(csr) {
		return nil
	}
	csr, err := s.sign(csr)
	if err != nil {
		return fmt.Errorf("error auto signing csr: %v", err)
	}
	_, err = s.client.Certificates().CertificateSigningRequests().UpdateStatus(csr)
	if err != nil {
		return fmt.Errorf("error updating signature for csr: %v", err)
	}
	return nil
}

// Sign will make an external call to GKE order to sign the given
// *capi.CertificateSigningRequest, using the GKESigner's
// kubeConfigFile.
func (s *GKESigner) sign(csr *capi.CertificateSigningRequest) (*capi.CertificateSigningRequest, error) {
	result := s.webhook.WithExponentialBackoff(func() rest.Result {
		return s.webhook.RestClient.Post().Body(csr).Do()
	})

	if err := result.Error(); err != nil {
		if bodyErr := s.resultBodyError(result); bodyErr != nil {
			return nil, s.webhookError(csr, bodyErr)
		}
		return nil, s.webhookError(csr, err)
	}

	var statusCode int
	if result.StatusCode(&statusCode); statusCode < 200 || statusCode >= 300 {
		return nil, s.webhookError(csr, fmt.Errorf("received unsuccessful response code from webhook: %d", statusCode))
	}

	result_csr := &capi.CertificateSigningRequest{}

	if err := result.Into(result_csr); err != nil {
		return nil, s.webhookError(result_csr, err)
	}

	// Keep the original CSR intact, and only update fields we expect to change.
	csr.Status.Certificate = result_csr.Status.Certificate
	return csr, nil
}

func (s *GKESigner) webhookError(csr *capi.CertificateSigningRequest, err error) error {
	glog.V(2).Infof("error contacting webhook backend: %s", err)
	s.recorder.Eventf(csr, "Warning", "SigningError", "error while calling GKE: %v", err)
	return err
}

// signResultError represents the structured response body of a failed call to
// GKE's SignCertificate API.
type signResultError struct {
	Error struct {
		Code    int
		Message string
		Status  string
	}
}

// resultBodyError attempts to extract an error out of a response body.
func (s *GKESigner) resultBodyError(result rest.Result) error {
	body, _ := result.Raw()
	var sre signResultError
	if err := json.Unmarshal(body, &sre); err == nil {
		return fmt.Errorf("server responded with error: %s", sre.Error.Message)
	}
	return nil
}
