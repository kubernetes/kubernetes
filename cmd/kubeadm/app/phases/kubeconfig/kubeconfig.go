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
	"io/ioutil"
	"os"
	"path/filepath"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
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
	caCert, caKey, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkiDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil || caCert == nil || caKey == nil {
		return fmt.Errorf("couldn't create a kubeconfig; the CA files couldn't be loaded: %v")
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
	filename := filepath.Join(outDir, fmt.Sprintf("%s.conf", client))
	return WriteKubeconfigToDisk(filename, config)
}

func WriteKubeconfigToDisk(filename string, kubeconfig *clientcmdapi.Config) error {
	// Convert the KubeConfig object to a byte array
	content, err := clientcmd.Write(*kubeconfig)
	if err != nil {
		return err
	}

	// Create the directory if it does not exist
	dir := filepath.Dir(filename)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err = os.MkdirAll(dir, KubernetesDirPermissions); err != nil {
			return err
		}
	}

	// Check if the file exist, and if it does check if the current content matches the expected content
	if _, err := os.Stat(filename); err == nil {
		// The kubeconfig already exists, let's check if it has got the right content
		fcc := preflight.FileContentCheck{
			Path:    filename,
			Content: content,
		}
		if _, errs := fcc.Check(); len(errs) > 0 {
			return fmt.Errorf("the kubeconfig file %q already exists, but has the wrong content", filename)
		} else {
			// There we're no errors; the file that is on disk is up-to-date
			fmt.Printf("[kubeconfig] KubeConfig file %q already exists and is up to date\n", filename)
			return nil
		}
	}

	// Write the file to disk
	if err := ioutil.WriteFile(filename, content, KubeConfigFilePermissions); err != nil {
		return err
	}

	fmt.Printf("[kubeconfig] Wrote KubeConfig file to disk: %q\n", filename)
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
