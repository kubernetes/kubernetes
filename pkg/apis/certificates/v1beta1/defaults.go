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

	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	certificates "k8s.io/kubernetes/pkg/apis/certificates"
)

func init() {
	localSchemeBuilder.Register(addDefaultingFuncs)
}

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

func IsKubeletServingCSR(req *x509.CertificateRequest, usages []certificatesv1beta1.KeyUsage) bool {
	return certificates.IsKubeletServingCSR(req, usagesToSet(usages))
}

func IsKubeletClientCSR(req *x509.CertificateRequest, usages []certificatesv1beta1.KeyUsage) bool {
	return certificates.IsKubeletClientCSR(req, usagesToSet(usages))
}

func usagesToSet(usages []certificatesv1beta1.KeyUsage) sets.String {
	result := sets.NewString()
	for _, usage := range usages {
		result.Insert(string(usage))
	}
	return result
}
