// Package kubernetesx509 contains routines for creating and parsing
// x509 certificates and certificate requests that embed Kubernetes-specific
// X.509 extensions that communicate the Kubernetes identity of a given
// workload.
package kubernetesx509

import (
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"fmt"
)

var (
	// The ASN.1 Private Enterprise Number assigned to the Kubernetes project.
	// Used to name X.509 extensions that communicate Kubernetes-specific
	// concepts.
	KubernetesPEN = asn1.ObjectIdentifier{1, 3, 6, 1, 4, 1, 57683}

	// OIDPodIdentity identifies the Kubernetes PodIdentity X.509 extension.
	OIDPodIdentity = makeKubernetesOID(1)
)

func makeKubernetesOID(subIDs ...int) asn1.ObjectIdentifier {
	base := asn1.ObjectIdentifier{}
	base = append(base, KubernetesPEN...)
	base = append(base, subIDs...)
	return base
}

type PodIdentity struct {
	Namespace          string
	ServiceAccountName string
	PodName            string
	PodUID             string
	NodeName           string
}

type podIdentityASN1 struct {
	Namespace          string `asn1:"utf8"`
	ServiceAccountName string `asn1:"utf8"`
	PodName            string `asn1:"utf8"`
	PodUID             string `asn1:"utf8"`
	NodeName           string `asn1:"utf8"`
}

// AddPodIdentityToCertificateRequest adds a pod identity to an
// x509.CertificateRequest.
//
// The certificate request can then be serialized, wrapped in PEM, and used in a
// Kubernetes CertificateSigningRequest object.
//
// Note that Kubernetes imposes special restrictions on
// CertificateSigningRequests that use the Kubernetes PodIdentity X.509
// extension, so at the present time only Kubelet will actually be able to
// create CertificateSigningRequest objects that use this extension.
func AddPodIdentityToCertificateRequest(pod *PodIdentity, req *x509.CertificateRequest) error {
	podIdentityBytes, err := asn1.Marshal(podIdentityASN1{
		Namespace:          pod.Namespace,
		ServiceAccountName: pod.ServiceAccountName,
		PodName:            pod.PodName,
		PodUID:             pod.PodUID,
		NodeName:           pod.NodeName,
	})
	if err != nil {
		return fmt.Errorf("while ASN1-marshaling PodIdentity extension: %w", err)
	}

	req.ExtraExtensions = append(req.ExtraExtensions, pkix.Extension{
		Id:    OIDPodIdentity,
		Value: podIdentityBytes,
	})

	return nil
}

// PodIdentityFromCertificateRequest extracts a pod identity from an
// x509.CertificateRequest, if it contains one.  Otherwise, it returns nil.
//
// Independent signers can use this routine to process CertificateSigningRequest
// objects that contain Kubernetes X.509 extensions.
//
// This routine does not validate the CSR signature; the client should call
// req.CheckSignature() before calling this function.
//
// This routine does not enforce the presence of any particular fields of the
// PodIdentity; for example, if the CSR does not contain Kubernetes
// namespace information, then the returned PodIdentity.Namespace will be
// the empty string.  The caller should validate the presence of the fields
// needed for their application.
func PodIdentityFromCertificateRequest(req *x509.CertificateRequest) (*PodIdentity, error) {
	podIdentityCount := 0

	podWire := podIdentityASN1{}
	for _, ext := range req.Extensions {
		switch {
		case ext.Id.Equal(OIDPodIdentity):
			podIdentityCount++

			_, err := asn1.Unmarshal(ext.Value, &podWire)
			if err != nil {
				return nil, fmt.Errorf("while unmarshaling Kubernetes PodIdentity extension: %w", err)
			}
		}
	}

	if podIdentityCount == 0 {
		return nil, nil
	}
	if podIdentityCount > 1 {
		return nil, fmt.Errorf("CertificateRequest contains multiple Kubernetes PodIdentity extensions")
	}

	pod := &PodIdentity{
		Namespace:          podWire.Namespace,
		ServiceAccountName: podWire.ServiceAccountName,
		PodName:            podWire.PodName,
		PodUID:             podWire.PodUID,
		NodeName:           podWire.NodeName,
	}

	return pod, nil
}

// AddPodIdentityToCertificate adds a Kubernetes PodIdentity extension to an
// x509.Certificate.
//
// Independent signers can use this routine to add PodIdentity extensions to the
// certificates they issue.  Note that it is not required, or even expected,
// that most independent signers will insert these extensions into their issued
// certificates.
func AddPodIdentityToCertificate(pod *PodIdentity, template *x509.Certificate) error {
	podIdentityBytes, err := asn1.Marshal(podIdentityASN1{
		Namespace:          pod.Namespace,
		ServiceAccountName: pod.ServiceAccountName,
		PodName:            pod.PodName,
		PodUID:             pod.PodUID,
		NodeName:           pod.NodeName,
	})
	if err != nil {
		return fmt.Errorf("while ASN1-marshaling PodIdentity extension: %w", err)
	}

	template.ExtraExtensions = append(template.ExtraExtensions, pkix.Extension{
		Id:    OIDPodIdentity,
		Value: podIdentityBytes,
	})

	return nil
}

// PodIdentityFromCertificate extracts a PodIdentity from a Certificate, if it
// contains one.  Otherwise it returns nil.
//
// This routine does not verify the Certificate; the client should call
// cert.Verify() before calling this function.
//
// This routine does not enforce the presence of any particular fields of the
// PodIdentity; for example, if the Certificate does not contain Kubernetes
// namespace information, then the returned PodIdentity.Namespace will be the
// empty string.  The caller should validate the presence of the fields needed
// for their application.
func PodIdentityFromCertificate(cert *x509.Certificate) (*PodIdentity, error) {
	podIdentityCount := 0

	podWire := podIdentityASN1{}
	for _, ext := range cert.Extensions {
		switch {
		case ext.Id.Equal(OIDPodIdentity):
			podIdentityCount++

			_, err := asn1.Unmarshal(ext.Value, &podWire)
			if err != nil {
				return nil, fmt.Errorf("while unmarshaling Kubernetes PodIdentity extension: %w", err)
			}
		}
	}

	if podIdentityCount == 0 {
		return nil, nil
	}
	if podIdentityCount > 1 {
		return nil, fmt.Errorf("certificate contains multiple Kubernetes PodIdentity extensions")
	}

	pod := &PodIdentity{
		Namespace:          podWire.Namespace,
		ServiceAccountName: podWire.ServiceAccountName,
		PodName:            podWire.PodName,
		PodUID:             podWire.PodUID,
		NodeName:           podWire.NodeName,
	}

	return pod, nil
}
