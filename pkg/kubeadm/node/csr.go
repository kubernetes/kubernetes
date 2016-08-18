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

package kubenode

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/certificates"
	unversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"

	//utilcertificates "k8s.io/kubernetes/pkg/util/certificates"
	"k8s.io/kubernetes/pkg/watch"
)

func getNodeName() string {
	return "TODO"
}

// runs on nodes
func PerformTLSBootstrap(params *kubeadmapi.BootstrapParams) (*clientcmdapi.Config, error) {
	// Create a restful client for doing the certificate signing request.
	pemData, err := ioutil.ReadFile(params.Discovery.CaCertFile)
	if err != nil {
		return nil, err
	}
	// TODO try all the api servers until we find one that works
	configBootstrap := kubeadmutil.CreateBasicClientConfig(
		"kubernetes", strings.Split(params.Discovery.ApiServerURLs, ",")[0], pemData,
	)

	nodeName := getNodeName()

	configBootstrap = kubeadmutil.MakeClientConfigWithToken(
		configBootstrap, "kubernetes", fmt.Sprintf("kubelet-%s", nodeName), params.Discovery.BearerToken,
	)
	clientConfig, err := clientcmd.NewDefaultClientConfig(
		*configBootstrap,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, err
	}

	client, err := unversionedcertificates.NewForConfig(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to create certificates signing request client: %v", err)
	}
	csrClient := client.CertificateSigningRequests()

	// Pass 'requestClientCertificate()' the CSR client, existing key data, and node name to
	// request for client certificate from the API server.
	certData, keyData, err := requestClientCertificate(csrClient, []byte{}, nodeName)
	if err != nil {
		return nil, fmt.Errorf("unable to request certificate from API server: %v", err)
	}
	// TODO transform clientcert into kubeconfig so that it can be written out on the node

	finalConfig := kubeadmutil.MakeClientConfigWithCerts(
		configBootstrap, "kubernetes", fmt.Sprintf("kubelet-%s", nodeName),
		keyData, certData,
	)

	return finalConfig, nil
}

// TODO: this function should be exported by kubelet package when the PR gets merged, so use that
func requestClientCertificate(client unversionedcertificates.CertificateSigningRequestInterface, existingKeyData []byte, nodeName string) (certData []byte, keyData []byte, err error) {
	subject := &pkix.Name{
		Organization: []string{"system:nodes"},
		CommonName:   fmt.Sprintf("system:node:%s", nodeName),
	}

	csr, keyData, err := newCertificateRequest(existingKeyData, subject, nil, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to generate certificate request: %v", err)
	}

	req, err := client.Create(&certificates.CertificateSigningRequest{
		TypeMeta:   unversioned.TypeMeta{Kind: "CertificateSigningRequest"},
		ObjectMeta: api.ObjectMeta{GenerateName: "csr-"},

		// Username, UID, Groups will be injected by API server.
		Spec: certificates.CertificateSigningRequestSpec{Request: csr},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("cannot create certificate signing request: %v", err)

	}

	// Make a default timeout = 3600s
	var defaultTimeoutSeconds int64 = 3600
	resultCh, err := client.Watch(api.ListOptions{
		Watch:          true,
		TimeoutSeconds: &defaultTimeoutSeconds,
		// Label and field selector are not used now.
	})
	if err != nil {
		return nil, nil, fmt.Errorf("cannot watch on the certificate signing request: %v", err)
	}

	var status certificates.CertificateSigningRequestStatus
	ch := resultCh.ResultChan()

	for {
		event, ok := <-ch
		if !ok {
			break
		}

		if event.Type == watch.Modified {
			if event.Object.(*certificates.CertificateSigningRequest).UID != req.UID {
				continue
			}
			status = event.Object.(*certificates.CertificateSigningRequest).Status
			for _, c := range status.Conditions {
				if c.Type == certificates.CertificateDenied {
					return nil, nil, fmt.Errorf("certificate signing request is not approved: %v, %v", c.Reason, c.Message)
				}
				if c.Type == certificates.CertificateApproved && status.Certificate != nil {
					return status.Certificate, keyData, nil
				}
			}
		}
	}

	return nil, nil, fmt.Errorf("watch channel closed")
}

// NewCertificateRequest generates a PEM-encoded CSR using the supplied private
// key data, subject, and SANs. If the private key data is empty, it generates a
// new ECDSA P256 key to use and returns it together with the CSR data.
func newCertificateRequest(keyData []byte, subject *pkix.Name, dnsSANs []string, ipSANs []net.IP) (csr []byte, key []byte, err error) {
	var privateKey interface{}
	var privateKeyPemBlock *pem.Block

	if len(keyData) == 0 {
		privateKey, err = ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
		if err != nil {
			return nil, nil, err
		}

		ecdsaKey := privateKey.(*ecdsa.PrivateKey)
		derBytes, err := x509.MarshalECPrivateKey(ecdsaKey)
		if err != nil {
			return nil, nil, err
		}

		privateKeyPemBlock = &pem.Block{
			Type:  "EC PRIVATE KEY",
			Bytes: derBytes,
		}
	} else {
		privateKeyPemBlock, _ = pem.Decode(keyData)
	}

	var sigType x509.SignatureAlgorithm

	switch privateKeyPemBlock.Type {
	case "EC PRIVATE KEY":
		privateKey, err = x509.ParseECPrivateKey(privateKeyPemBlock.Bytes)
		if err != nil {
			return nil, nil, err
		}
		ecdsaKey := privateKey.(*ecdsa.PrivateKey)
		switch ecdsaKey.Curve.Params().BitSize {
		case 512:
			sigType = x509.ECDSAWithSHA512
		case 384:
			sigType = x509.ECDSAWithSHA384
		default:
			sigType = x509.ECDSAWithSHA256
		}
	case "RSA PRIVATE KEY":
		privateKey, err = x509.ParsePKCS1PrivateKey(privateKeyPemBlock.Bytes)
		if err != nil {
			return nil, nil, err
		}
		rsaKey := privateKey.(*rsa.PrivateKey)
		keySize := rsaKey.N.BitLen()
		switch {
		case keySize >= 4096:
			sigType = x509.SHA512WithRSA
		case keySize >= 3072:
			sigType = x509.SHA384WithRSA
		default:
			sigType = x509.SHA256WithRSA
		}
	default:
		return nil, nil, fmt.Errorf("unsupported key type: %s", privateKeyPemBlock.Type)
	}

	template := &x509.CertificateRequest{
		Subject:            *subject,
		SignatureAlgorithm: sigType,
		DNSNames:           dnsSANs,
		IPAddresses:        ipSANs,
	}

	csr, err = x509.CreateCertificateRequest(cryptorand.Reader, template, privateKey)
	if err != nil {
		return nil, nil, err
	}

	csrPemBlock := &pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csr,
	}

	return pem.EncodeToMemory(csrPemBlock), pem.EncodeToMemory(privateKeyPemBlock), nil
}
