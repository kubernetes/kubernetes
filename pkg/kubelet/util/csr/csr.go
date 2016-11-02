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

package csr

import (
	"crypto/x509/pkix"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/certificates"
	unversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/types"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/pkg/watch"
)

// RequestNodeCertificate will create a certificate signing request and send it to API server,
// then it will watch the object's status, once approved by API server, it will return the API
// server's issued certificate (pem-encoded). If there is any errors, or the watch timeouts,
// it will return an error. This is intended for use on nodes (kubelet and kubeadm).
func RequestNodeCertificate(client unversionedcertificates.CertificateSigningRequestInterface, privateKeyData []byte, nodeName types.NodeName) (certData []byte, err error) {
	subject := &pkix.Name{
		Organization: []string{"system:nodes"},
		CommonName:   fmt.Sprintf("system:node:%s", nodeName),
	}

	privateKey, err := certutil.ParsePrivateKeyPEM(privateKeyData)
	if err != nil {
		return nil, fmt.Errorf("invalid private key for certificate request: %v", err)
	}
	csr, err := certutil.MakeCSR(privateKey, subject, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("unable to generate certificate request: %v", err)
	}

	req, err := client.Create(&certificates.CertificateSigningRequest{
		// Username, UID, Groups will be injected by API server.
		TypeMeta:   unversioned.TypeMeta{Kind: "CertificateSigningRequest"},
		ObjectMeta: api.ObjectMeta{GenerateName: "csr-"},

		// TODO: For now, this is a request for a certificate with allowed usage of "TLS Web Client Authentication".
		// Need to figure out whether/how to surface the allowed usage in the spec.
		Spec: certificates.CertificateSigningRequestSpec{Request: csr},
	})
	if err != nil {
		return nil, fmt.Errorf("cannot create certificate signing request: %v", err)

	}

	// Make a default timeout = 3600s.
	var defaultTimeoutSeconds int64 = 3600
	resultCh, err := client.Watch(api.ListOptions{
		Watch:          true,
		TimeoutSeconds: &defaultTimeoutSeconds,
		FieldSelector:  fields.OneTermEqualSelector("metadata.name", req.Name),
	})
	if err != nil {
		return nil, fmt.Errorf("cannot watch on the certificate signing request: %v", err)
	}

	var status certificates.CertificateSigningRequestStatus
	ch := resultCh.ResultChan()

	for {
		event, ok := <-ch
		if !ok {
			break
		}

		if event.Type == watch.Modified || event.Type == watch.Added {
			if event.Object.(*certificates.CertificateSigningRequest).UID != req.UID {
				continue
			}
			status = event.Object.(*certificates.CertificateSigningRequest).Status
			for _, c := range status.Conditions {
				if c.Type == certificates.CertificateDenied {
					return nil, fmt.Errorf("certificate signing request is not approved, reason: %v, message: %v", c.Reason, c.Message)
				}
				if c.Type == certificates.CertificateApproved && status.Certificate != nil {
					return status.Certificate, nil
				}
			}
		}
	}

	return nil, fmt.Errorf("watch channel closed")
}
