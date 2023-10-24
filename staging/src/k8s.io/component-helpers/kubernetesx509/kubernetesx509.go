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
	ServiceAccountUID  string
	PodName            string
	PodUID             string
	NodeName           string
	NodeUID            string
}

type podIdentityASN1 struct {
	Namespace          string `asn1:"utf8"`
	ServiceAccountName string `asn1:"utf8"`
	ServiceAccountUID  string `asn1:"utf8"`
	PodName            string `asn1:"utf8"`
	PodUID             string `asn1:"utf8"`
	NodeName           string `asn1:"utf8"`
	NodeUID            string `asn1:"utf8"`
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
		ServiceAccountUID:  pod.ServiceAccountUID,
		PodName:            pod.PodName,
		PodUID:             pod.PodUID,
		NodeName:           pod.NodeName,
		NodeUID:            pod.NodeUID,
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
		ServiceAccountUID:  podWire.ServiceAccountUID,
		PodName:            podWire.PodName,
		PodUID:             podWire.PodUID,
		NodeName:           podWire.NodeName,
		NodeUID:            podWire.NodeUID,
	}

	return pod, nil
}
