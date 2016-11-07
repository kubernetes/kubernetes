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

package node

import (
	"crypto/x509/pkix"
	"fmt"

	certclient "k8s.io/client-go/kubernetes/typed/certificates/v1alpha1"
	"k8s.io/client-go/pkg/api/unversioned"
	api "k8s.io/client-go/pkg/api/v1"
	certificates "k8s.io/client-go/pkg/apis/certificates/v1alpha1"
	"k8s.io/client-go/pkg/fields"
	"k8s.io/client-go/pkg/types"
	certutil "k8s.io/client-go/pkg/util/cert"
	"k8s.io/client-go/pkg/watch"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// PerformTLSBootstrap executes a certificate signing request with the
// provided connection details.
func PerformTLSBootstrap(connection *ConnectionDetails) (*clientcmdapi.Config, error) {
	csrClient := connection.CertClient.CertificateSigningRequests()

	fmt.Println("<node/csr> created API client to obtain unique certificate for this node, generating keys and certificate signing request")

	key, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to generating private key [%v]", err)
	}
	cert, err := RequestNodeCertificate(csrClient, key, connection.NodeName)
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to request signed certificate from the API server [%v]", err)
	}
	fmtCert, err := certutil.FormatBytesCert(cert)
	if err != nil {
		return nil, fmt.Errorf("<node/csr> failed to format certificate [%v]", err)
	}
	fmt.Printf("<node/csr> received signed certificate from the API server:\n%s\n", fmtCert)
	fmt.Println("<node/csr> generating kubelet configuration")

	bareClientConfig := kubeadmutil.CreateBasicClientConfig("kubernetes", connection.Endpoint, connection.CACert)
	finalConfig := kubeadmutil.MakeClientConfigWithCerts(
		bareClientConfig, "kubernetes", fmt.Sprintf("kubelet-%s", connection.NodeName),
		key, cert,
	)

	return finalConfig, nil
}

func RequestNodeCertificate(client certclient.CertificateSigningRequestInterface, privateKeyData []byte, nodeName types.NodeName) (certData []byte, err error) {
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
		FieldSelector:  fields.OneTermEqualSelector("metadata.name", req.Name).String(),
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
