/*
Copyright 2018 The Kubernetes Authors.

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

package renewal

import (
	"crypto"
	"crypto/x509"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

// FileRenewer define a certificate renewer implementation that uses given CA cert and key for generating new certificates
type FileRenewer struct {
	caCert *x509.Certificate
	caKey  crypto.Signer
}

// NewFileRenewer returns a new certificate renewer that uses given CA cert and key for generating new certificates
func NewFileRenewer(caCert *x509.Certificate, caKey crypto.Signer) *FileRenewer {
	return &FileRenewer{
		caCert: caCert,
		caKey:  caKey,
	}
}

// Renew a certificate using a given CA cert and key
func (r *FileRenewer) Renew(cfg *pkiutil.CertConfig) (*x509.Certificate, crypto.Signer, error) {
	return pkiutil.NewCertAndKey(r.caCert, r.caKey, cfg)
}
