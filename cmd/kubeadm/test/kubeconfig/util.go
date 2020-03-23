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

package kubeconfig

import (
	"crypto/x509"
	"encoding/pem"
	"testing"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
)

// AssertKubeConfigCurrentCluster is a utility function for kubeadm testing that asserts if the CurrentCluster in
// the given KubeConfig object contains refers to a specific cluster
func AssertKubeConfigCurrentCluster(t *testing.T, config *clientcmdapi.Config, expectedAPIServerAddress string, expectedAPIServerCaCert *x509.Certificate) {
	currentContext := config.Contexts[config.CurrentContext]
	currentCluster := config.Clusters[currentContext.Cluster]

	// Assert expectedAPIServerAddress
	if currentCluster.Server != expectedAPIServerAddress {
		t.Errorf("kubeconfig.currentCluster.Server is [%s], expected [%s]", currentCluster.Server, expectedAPIServerAddress)
	}

	// Assert the APIServerCaCert
	if len(currentCluster.CertificateAuthorityData) == 0 {
		t.Error("kubeconfig.currentCluster.CertificateAuthorityData is empty, expected not empty")
		return
	}

	block, _ := pem.Decode(currentCluster.CertificateAuthorityData)
	currentAPIServerCaCert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Errorf("kubeconfig.currentCluster.CertificateAuthorityData is not a valid CA: %v", err)
		return
	}

	if !currentAPIServerCaCert.Equal(expectedAPIServerCaCert) {
		t.Errorf("kubeconfig.currentCluster.CertificateAuthorityData not correspond to the expected CA cert")
	}
}

// AssertKubeConfigCurrentAuthInfoWithClientCert is a utility function for kubeadm testing that asserts if the CurrentAuthInfo in
// the given KubeConfig object contains a clientCert that refers to a specific client name, is signed by the expected CA, includes the expected organizations
func AssertKubeConfigCurrentAuthInfoWithClientCert(t *testing.T, config *clientcmdapi.Config, signinCa *x509.Certificate, expectedClientName string, expectedOrganizations ...string) {
	currentContext := config.Contexts[config.CurrentContext]
	currentAuthInfo := config.AuthInfos[currentContext.AuthInfo]

	// assert clientCert
	if len(currentAuthInfo.ClientCertificateData) == 0 {
		t.Error("kubeconfig.currentAuthInfo.ClientCertificateData is empty, expected not empty")
		return
	}

	block, _ := pem.Decode(config.AuthInfos[currentContext.AuthInfo].ClientCertificateData)
	currentClientCert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Errorf("kubeconfig.currentAuthInfo.ClientCertificateData is not a valid CA: %v", err)
		return
	}

	// Asserts the clientCert is signed by the signinCa
	certstestutil.AssertCertificateIsSignedByCa(t, currentClientCert, signinCa)

	// Asserts the clientCert has ClientAuth ExtKeyUsage
	certstestutil.AssertCertificateHasClientAuthUsage(t, currentClientCert)

	// Asserts the clientCert has expected expectedUserName as CommonName
	certstestutil.AssertCertificateHasCommonName(t, currentClientCert, expectedClientName)

	// Asserts the clientCert has expected Organizations
	certstestutil.AssertCertificateHasOrganizations(t, currentClientCert, expectedOrganizations...)
}

// AssertKubeConfigCurrentAuthInfoWithToken is a utility function for kubeadm testing that asserts if the CurrentAuthInfo in
// the given KubeConfig object refers to expected token
func AssertKubeConfigCurrentAuthInfoWithToken(t *testing.T, config *clientcmdapi.Config, expectedClientName, expectedToken string) {
	currentContext := config.Contexts[config.CurrentContext]
	currentAuthInfo := config.AuthInfos[currentContext.AuthInfo]

	// assert token
	if currentAuthInfo.Token != expectedToken {
		t.Errorf("kubeconfig.currentAuthInfo.Token [%s], expected [%s]", currentAuthInfo.Token, expectedToken)
		return
	}
}
