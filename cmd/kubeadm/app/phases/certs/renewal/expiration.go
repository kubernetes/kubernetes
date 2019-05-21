/*
Copyright 2019 The Kubernetes Authors.

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
	"crypto/x509"
	"time"
)

// ExpirationInfo defines expiration info for a certificate
type ExpirationInfo struct {
	// Name of the certificate
	// For PKI certificates, it is the name defined in the certsphase package, while for certificates
	// embedded in the kubeConfig files, it is the kubeConfig file name defined in the kubeadm constants package.
	// If you use the CertificateRenewHandler returned by Certificates func, handler.Name already contains the right value.
	Name string

	// ExpirationDate defines certificate expiration date
	ExpirationDate time.Time

	// ExternallyManaged defines if the certificate is externally managed, that is when
	// the signing CA certificate is provided without the certificate key (In this case kubeadm can't renew the certificate)
	ExternallyManaged bool
}

// newExpirationInfo returns a new ExpirationInfo
func newExpirationInfo(name string, cert *x509.Certificate, externallyManaged bool) *ExpirationInfo {
	return &ExpirationInfo{
		Name:              name,
		ExpirationDate:    cert.NotAfter,
		ExternallyManaged: externallyManaged,
	}
}

// ResidualTime returns the time missing to expiration
func (e *ExpirationInfo) ResidualTime() time.Duration {
	return e.ExpirationDate.Sub(time.Now())
}
