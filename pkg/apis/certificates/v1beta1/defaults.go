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

package v1beta1

import (
	"crypto/x509"
	"fmt"
	"reflect"
	"strings"

	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_CertificateSigningRequestSpec(obj *certificatesv1beta1.CertificateSigningRequestSpec) {
	if obj.Usages == nil {
		obj.Usages = []certificatesv1beta1.KeyUsage{certificatesv1beta1.UsageDigitalSignature, certificatesv1beta1.UsageKeyEncipherment}
	}

	if obj.SignerName == nil {
		signerName := DefaultSignerNameFromSpec(obj)
		obj.SignerName = &signerName
	}
}

func SetDefaults_CertificateSigningRequestCondition(obj *certificatesv1beta1.CertificateSigningRequestCondition) {
	if len(obj.Status) == 0 {
		obj.Status = v1.ConditionTrue
	}
}

// DefaultSignerNameFromSpec will determine the signerName that should be set
// by attempting to inspect the 'request' content and the spec options.
func DefaultSignerNameFromSpec(obj *certificatesv1beta1.CertificateSigningRequestSpec) string {
	csr, err := ParseCSR(obj.Request)
	switch {
	case err != nil:
		// Set the signerName to 'legacy-unknown' as the CSR could not be
		// recognised.
		return certificatesv1beta1.LegacyUnknownSignerName
	case IsKubeletClientCSR(csr, obj.Usages):
		return certificatesv1beta1.KubeAPIServerClientKubeletSignerName
	case IsKubeletServingCSR(csr, obj.Usages):
		return certificatesv1beta1.KubeletServingSignerName
	default:
		return certificatesv1beta1.LegacyUnknownSignerName
	}
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

func IsKubeletServingCSR(req *x509.CertificateRequest, usages []certificatesv1beta1.KeyUsage) bool {
	return ValidateKubeletServingCSR(req, usages) == nil
}
func ValidateKubeletServingCSR(req *x509.CertificateRequest, usages []certificatesv1beta1.KeyUsage) error {
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

	requiredUsages := []certificatesv1beta1.KeyUsage{
		certificatesv1beta1.UsageDigitalSignature,
		certificatesv1beta1.UsageKeyEncipherment,
		certificatesv1beta1.UsageServerAuth,
	}
	if !equalUnsorted(requiredUsages, usages) {
		return fmt.Errorf("usages did not match %v", requiredUsages)
	}

	if !strings.HasPrefix(req.Subject.CommonName, "system:node:") {
		return commonNameNotSystemNode
	}

	return nil
}

func IsKubeletClientCSR(req *x509.CertificateRequest, usages []certificatesv1beta1.KeyUsage) bool {
	return ValidateKubeletClientCSR(req, usages) == nil
}
func ValidateKubeletClientCSR(req *x509.CertificateRequest, usages []certificatesv1beta1.KeyUsage) error {
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

	requiredUsages := []certificatesv1beta1.KeyUsage{
		certificatesv1beta1.UsageDigitalSignature,
		certificatesv1beta1.UsageKeyEncipherment,
		certificatesv1beta1.UsageClientAuth,
	}
	if !equalUnsorted(requiredUsages, usages) {
		return fmt.Errorf("usages did not match %v", requiredUsages)
	}

	return nil
}

// equalUnsorted compares two []string for equality of contents regardless of
// the order of the elements
func equalUnsorted(left, right []certificatesv1beta1.KeyUsage) bool {
	l := sets.NewString()
	for _, s := range left {
		l.Insert(string(s))
	}
	r := sets.NewString()
	for _, s := range right {
		r.Insert(string(s))
	}
	return l.Equal(r)
}
