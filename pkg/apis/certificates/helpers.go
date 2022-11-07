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

package certificates

import (
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
)

// ParseCSR extracts the CSR from the bytes and decodes it.
func ParseCSR(pemBytes []byte) (*x509.CertificateRequest, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil || block.Type != "CERTIFICATE REQUEST" {
		return nil, errors.New("PEM block type must be CERTIFICATE REQUEST")
	}
	csr, err := x509.ParseCertificateRequest(block.Bytes)
	if err != nil {
		return nil, err
	}
	return csr, nil
}

var (
	organizationNotSystemNodesErr = fmt.Errorf("subject organization is not system:nodes")
	commonNameNotSystemNode       = fmt.Errorf("subject common name does not begin with system:node:")
	dnsOrIPSANRequiredErr         = fmt.Errorf("DNS or IP subjectAltName is required")
	dnsSANNotAllowedErr           = fmt.Errorf("DNS subjectAltNames are not allowed")
	emailSANNotAllowedErr         = fmt.Errorf("Email subjectAltNames are not allowed")
	ipSANNotAllowedErr            = fmt.Errorf("IP subjectAltNames are not allowed")
	uriSANNotAllowedErr           = fmt.Errorf("URI subjectAltNames are not allowed")
)

var (
	kubeletServingRequiredUsages = sets.NewString(
		string(UsageDigitalSignature),
		string(UsageKeyEncipherment),
		string(UsageServerAuth),
	)
	kubeletServingRequiredUsagesNoRSA = sets.NewString(
		string(UsageDigitalSignature),
		string(UsageServerAuth),
	)
)

func IsKubeletServingCSR(req *x509.CertificateRequest, usages sets.String) bool {
	return ValidateKubeletServingCSR(req, usages) == nil
}
func ValidateKubeletServingCSR(req *x509.CertificateRequest, usages sets.String) error {
	if !reflect.DeepEqual([]string{"system:nodes"}, req.Subject.Organization) {
		return organizationNotSystemNodesErr
	}

	// at least one of dnsNames or ipAddresses must be specified
	if len(req.DNSNames) == 0 && len(req.IPAddresses) == 0 {
		return dnsOrIPSANRequiredErr
	}

	if len(req.EmailAddresses) > 0 {
		return emailSANNotAllowedErr
	}
	if len(req.URIs) > 0 {
		return uriSANNotAllowedErr
	}

	if !kubeletServingRequiredUsages.Equal(usages) && !kubeletServingRequiredUsagesNoRSA.Equal(usages) {
		return fmt.Errorf("usages did not match %v", kubeletServingRequiredUsages.List())
	}

	if !strings.HasPrefix(req.Subject.CommonName, "system:node:") {
		return commonNameNotSystemNode
	}

	return nil
}

var (
	kubeletClientRequiredUsagesNoRSA = sets.NewString(
		string(UsageDigitalSignature),
		string(UsageClientAuth),
	)
	kubeletClientRequiredUsages = sets.NewString(
		string(UsageDigitalSignature),
		string(UsageKeyEncipherment),
		string(UsageClientAuth),
	)
)

func IsKubeletClientCSR(req *x509.CertificateRequest, usages sets.String) bool {
	return ValidateKubeletClientCSR(req, usages) == nil
}
func ValidateKubeletClientCSR(req *x509.CertificateRequest, usages sets.String) error {
	if !reflect.DeepEqual([]string{"system:nodes"}, req.Subject.Organization) {
		return organizationNotSystemNodesErr
	}

	if len(req.DNSNames) > 0 {
		return dnsSANNotAllowedErr
	}
	if len(req.EmailAddresses) > 0 {
		return emailSANNotAllowedErr
	}
	if len(req.IPAddresses) > 0 {
		return ipSANNotAllowedErr
	}
	if len(req.URIs) > 0 {
		return uriSANNotAllowedErr
	}

	if !strings.HasPrefix(req.Subject.CommonName, "system:node:") {
		return commonNameNotSystemNode
	}

	if !kubeletClientRequiredUsages.Equal(usages) && !kubeletClientRequiredUsagesNoRSA.Equal(usages) {
		return fmt.Errorf("usages did not match %v", kubeletClientRequiredUsages.List())
	}

	return nil
}
