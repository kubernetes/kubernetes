/*
Copyright 2020 The Kubernetes Authors.

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

package v1

import (
	"encoding/base64"
	"fmt"

	certificatesv1 "k8s.io/api/certificates/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add field conversion funcs.
	err := scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("CertificateSigningRequest"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name",
				"spec.signerName":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		},
	)
	if err != nil {
		return err
	}

	err = scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("PodCertificateRequest"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", "spec.signerName", "spec.podName", "spec.nodeName":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		},
	)
	if err != nil {
		return fmt.Errorf("while adding PodCertificateRequest field label conversion func: %w", err)
	}

	return nil
}

func Convert_v1_PodCertificateRequestSpec_To_certificates_PodCertificateRequestSpec(in *certificatesv1.PodCertificateRequestSpec, out *certificates.PodCertificateRequestSpec, s conversion.Scope) error {
	out.SignerName = in.SignerName
	out.PodName = in.PodName
	out.PodUID = in.PodUID
	out.ServiceAccountName = in.ServiceAccountName
	out.ServiceAccountUID = in.ServiceAccountUID
	out.NodeName = in.NodeName
	out.NodeUID = in.NodeUID
	out.MaxExpirationSeconds = in.MaxExpirationSeconds
	out.StubPKCS10Request = in.StubPKCS10Request
	out.UnverifiedUserAnnotations = in.UnverifiedUserAnnotations
	out.PKIXPublicKey = nil
	out.ProofOfPossession = nil
	return nil
}

func Convert_certificates_PodCertificateRequestSpec_To_v1_PodCertificateRequestSpec(in *certificates.PodCertificateRequestSpec, out *certificatesv1.PodCertificateRequestSpec, s conversion.Scope) error {
	out.SignerName = in.SignerName
	out.PodName = in.PodName
	out.PodUID = in.PodUID
	out.ServiceAccountName = in.ServiceAccountName
	out.ServiceAccountUID = in.ServiceAccountUID
	out.NodeName = in.NodeName
	out.NodeUID = in.NodeUID
	out.MaxExpirationSeconds = in.MaxExpirationSeconds
	out.StubPKCS10Request = in.StubPKCS10Request
	out.UnverifiedUserAnnotations = in.UnverifiedUserAnnotations
	return nil
}

func Convert_v1_PodCertificateRequest_To_certificates_PodCertificateRequest(in *certificatesv1.PodCertificateRequest, out *certificates.PodCertificateRequest, s conversion.Scope) error {
	if err := autoConvert_v1_PodCertificateRequest_To_certificates_PodCertificateRequest(in, out, s); err != nil {
		return err
	}

	if out.Annotations != nil {
		if pkixBase64, ok := out.Annotations["certificates.k8s.io/initial-pkix-public-key"]; ok {
			pkix, err := base64.StdEncoding.DecodeString(pkixBase64)
			if err == nil {
				out.Spec.PKIXPublicKey = pkix
			}
			out.Annotations = deepCopyStringMap(out.Annotations)
			delete(out.Annotations, "certificates.k8s.io/initial-pkix-public-key")
		}
		if popBase64, ok := out.Annotations["certificates.k8s.io/initial-proof-of-possession"]; ok {
			pop, err := base64.StdEncoding.DecodeString(popBase64)
			if err == nil {
				out.Spec.ProofOfPossession = pop
			}
			out.Annotations = deepCopyStringMap(out.Annotations)
			delete(out.Annotations, "certificates.k8s.io/initial-proof-of-possession")
		}
		if len(out.Annotations) == 0 {
			out.Annotations = nil
		}
	}
	return nil
}

func Convert_certificates_PodCertificateRequest_To_v1_PodCertificateRequest(in *certificates.PodCertificateRequest, out *certificatesv1.PodCertificateRequest, s conversion.Scope) error {
	if err := autoConvert_certificates_PodCertificateRequest_To_v1_PodCertificateRequest(in, out, s); err != nil {
		return err
	}

	if len(in.Spec.PKIXPublicKey) > 0 {
		out.Annotations = deepCopyStringMap(out.Annotations)
		if out.Annotations == nil {
			out.Annotations = map[string]string{}
		}
		out.Annotations["certificates.k8s.io/initial-pkix-public-key"] = base64.StdEncoding.EncodeToString(in.Spec.PKIXPublicKey)
	}
	if len(in.Spec.ProofOfPossession) > 0 {
		out.Annotations = deepCopyStringMap(out.Annotations)
		if out.Annotations == nil {
			out.Annotations = map[string]string{}
		}
		out.Annotations["certificates.k8s.io/initial-proof-of-possession"] = base64.StdEncoding.EncodeToString(in.Spec.ProofOfPossession)
	}
	return nil
}

func deepCopyStringMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	ret := make(map[string]string, len(m))
	for k, v := range m {
		ret[k] = v
	}
	return ret
}
