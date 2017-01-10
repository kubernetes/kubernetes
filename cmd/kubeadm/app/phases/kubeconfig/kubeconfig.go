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

package kubeconfig

import (
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"os"
	"path"

	certconstants "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

const (
	KubernetesDirPermissions  = 0700
	KubeConfigFilePermissions = 0600
	AdminKubeConfigFileName   = "admin.conf"
	KubeletKubeConfigFileName = "kubelet.conf"
)

// This function is called from the main init and does the work for the default phase behaviour
// TODO: Make an integration test for this function that runs after the certificates phase
// and makes sure that those two phases work well together...
func CreateAdminAndKubeletKubeConfig(masterEndpoint, pkiDir, outDir string) error {

	// Try to load ca.crt and ca.key from the PKI directory
	caCert, caKey, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, certconstants.CACertAndKeyBaseName)
	if err != nil {
		return fmt.Errorf("couldn't create a kubeconfig; the CA files couldn't be parsed: %v")
	}

	// User admin should have full access to the cluster
	if err := createKubeConfigFileForClient(masterEndpoint, "admin", outDir, caCert, caKey); err != nil {
		return fmt.Errorf("couldn't create a kubeconfig file for admin: %v", err)
	}

	// TODO: The kubelet should have limited access to the cluster. Right now, this gives kubelet basically root access
	// and we do need that in the bootstrap phase, but we should swap it out after the control plane is up
	if err := createKubeConfigFileForClient(masterEndpoint, "kubelet", outDir, caCert, caKey); err != nil {
		return fmt.Errorf("couldn't create a kubeconfig file for kubelet: %v", err)
	}

	return nil
}

func createKubeConfigFileForClient(masterEndpoint, client, outDir string, caCert *x509.Certificate, caKey *rsa.PrivateKey) error {
	cert, key, err := pkiutil.NewClientKeyAndCert(caCert, caKey)
	if err != nil {
		return fmt.Errorf("failure while creating %s client certificate [%v]", client, err)
	}

	config := MakeClientConfigWithCerts(
		masterEndpoint,
		"kubernetes",
		client,
		certutil.EncodeCertPEM(caCert),
		certutil.EncodePrivateKeyPEM(key),
		certutil.EncodeCertPEM(cert),
	)

	// Write it now to a file
	filepath := path.Join(outDir, fmt.Sprintf("%s.conf", client))
	return WriteKubeconfigToDisk(filepath, config)
}

func WriteKubeconfigToDisk(filepath string, kubeconfig *clientcmdapi.Config) error {
	// If err == nil, the file exists. Oops, we don't allow the file to exist already, fail.
	// TODO: Should we allow overwriting a kubeconfig file that does already exist?
	if _, err := os.Stat(filepath); err == nil {
		return fmt.Errorf("kubeconfig file %s already exists, but must not exist.", filepath)
	}

	if err := clientcmd.WriteToFileWithPermissions(*kubeconfig, filepath, KubernetesDirPermissions, KubeConfigFilePermissions); err != nil {
		return fmt.Errorf("failed to write to %q [%v]", filepath, err)
	}

	fmt.Printf("[kubeconfig] Wrote KubeConfig file to disk: %q\n", filepath)
	return nil
}

func createBasicClientConfig(serverURL string, clusterName string, userName string, caCert []byte) *clientcmdapi.Config {
	config := clientcmdapi.NewConfig()

	// Make a new cluster, specify the endpoint we'd like to talk to and the ca cert we're gonna use
	cluster := clientcmdapi.NewCluster()
	cluster.Server = serverURL
	cluster.CertificateAuthorityData = caCert

	// Specify a context where we're using that cluster and the username as the auth information
	contextName := fmt.Sprintf("%s@%s", userName, clusterName)
	context := clientcmdapi.NewContext()
	context.Cluster = clusterName
	context.AuthInfo = userName

	// Lastly, apply the created objects above to the config
	config.Clusters[clusterName] = cluster
	config.Contexts[contextName] = context
	config.CurrentContext = contextName
	return config
}

// Creates a clientcmdapi.Config object with access to the API server with client certificates
func MakeClientConfigWithCerts(serverURL, clusterName, userName string, caCert []byte, clientKey []byte, clientCert []byte) *clientcmdapi.Config {
	config := createBasicClientConfig(serverURL, clusterName, userName, caCert)

	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.ClientKeyData = clientKey
	authInfo.ClientCertificateData = clientCert

	config.AuthInfos[userName] = authInfo
	return config
}

// Creates a clientcmdapi.Config object with access to the API server with a token
func MakeClientConfigWithToken(serverURL, clusterName, userName string, caCert []byte, token string) *clientcmdapi.Config {
	config := createBasicClientConfig(serverURL, clusterName, userName, caCert)

	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.Token = token

	config.AuthInfos[userName] = authInfo
	return config
}
