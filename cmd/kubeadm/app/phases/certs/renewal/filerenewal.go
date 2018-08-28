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
	"crypto/rsa"
	"crypto/x509"

	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
)

// FileRenewal renews a certificate using local certs
type FileRenewal struct {
	caCert *x509.Certificate
	caKey  *rsa.PrivateKey
}

// NewFileRenewal takes a certificate pair to construct the Interface.
func NewFileRenewal(caCert *x509.Certificate, caKey *rsa.PrivateKey) Interface {
	return &FileRenewal{
		caCert: caCert,
		caKey:  caKey,
	}
}

// Renew takes a certificate using the cert and key
func (r *FileRenewal) Renew(cfg *certutil.Config) (*x509.Certificate, *rsa.PrivateKey, error) {
	return pkiutil.NewCertAndKey(r.caCert, r.caKey, cfg)
}
