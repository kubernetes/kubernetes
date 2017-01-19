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
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"os"
	"path"

	certutil "k8s.io/client-go/pkg/util/cert"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

const (
	KubernetesDirPermissions    = 0700
	AdminKubeConfigFileName     = "admin.conf"
	AdminKubeConfigClientName   = "kubernetes-admin"
	KubeletKubeConfigFileName   = "kubelet.conf"
	KubeletKubeConfigClientName = "kubelet"
)

// This function is called from the main init and does the work for the default phase behaviour
// TODO: Make an integration test for this function that runs after the certificates phase
// and makes sure that those two phases work well together...
func CreateAdminAndKubeletKubeConfig(masterEndpoint, pkiDir, outDir string) error {
	// Parse the certificate from a file
	caCertPath := path.Join(pkiDir, "ca.pem")
	caCerts, err := certutil.CertsFromFile(caCertPath)
	if err != nil {
		return fmt.Errorf("couldn't load the CA cert file %s: %v", caCertPath, err)
	}
	// We are only putting one certificate in the CA certificate pem file, so it's safe to just use the first one
	caCert := caCerts[0]

	// Parse the rsa private key from a file
	caKeyPath := path.Join(pkiDir, "ca-key.pem")
	priv, err := certutil.PrivateKeyFromFile(caKeyPath)
	if err != nil {
		return fmt.Errorf("couldn't load the CA private key file %s: %v", caKeyPath, err)
	}
	var caKey *rsa.PrivateKey
	switch k := priv.(type) {
	case *rsa.PrivateKey:
		caKey = k
	case *ecdsa.PrivateKey:
		// TODO: Abstract rsa.PrivateKey away and make certutil.NewSignedCert accept a ecdsa.PrivateKey as well
		// After that, we can support generating kubeconfig files from ecdsa private keys as well
		return fmt.Errorf("the CA private key file %s isn't in RSA format", caKeyPath)
	default:
		return fmt.Errorf("the CA private key file %s isn't in RSA format", caKeyPath)
	}

	// User admin should have full access to the cluster
	adminCertConfig := &certutil.Config{
		CommonName:   AdminKubeConfigClientName,
		Organization: []string{"system:masters"},
	}
	adminKubeConfigFilePath := path.Join(outDir, AdminKubeConfigFileName)
	if err := createKubeConfigFileForClient(masterEndpoint, adminKubeConfigFilePath, adminCertConfig, caCert, caKey); err != nil {
		return fmt.Errorf("couldn't create config for %s: %v", AdminKubeConfigClientName, err)
	}

	// The kubelet should have limited access to the cluster
	kubeletCertConfig := &certutil.Config{
		CommonName:   KubeletKubeConfigClientName,
		Organization: []string{"system:nodes"},
	}
	kubeletKubeConfigFilePath := path.Join(outDir, KubeletKubeConfigFileName)
	if err := createKubeConfigFileForClient(masterEndpoint, kubeletKubeConfigFilePath, kubeletCertConfig, caCert, caKey); err != nil {
		return fmt.Errorf("couldn't create config for %s: %v", KubeletKubeConfigClientName, err)
	}

	// TODO make credentials for the controller manager and kube proxy

	return nil
}

func createKubeConfigFileForClient(masterEndpoint, kubeConfigFilePath string, config *certutil.Config, caCert *x509.Certificate, caKey *rsa.PrivateKey) error {
	key, cert, err := certphase.NewClientKeyAndCert(config, caCert, caKey)
	if err != nil {
		return fmt.Errorf("failure while creating %s client certificate [%v]", config.CommonName, err)
	}

	kubeConfig := MakeClientConfigWithCerts(
		masterEndpoint,
		"kubernetes",
		config.CommonName,
		certutil.EncodeCertPEM(caCert),
		certutil.EncodePrivateKeyPEM(key),
		certutil.EncodeCertPEM(cert),
	)

	// Write it now to a file
	return WriteKubeconfigToDisk(kubeConfigFilePath, kubeConfig)
}

func WriteKubeconfigToDisk(filepath string, kubeconfig *clientcmdapi.Config) error {
	// Make sure the dir exists or can be created
	if err := os.MkdirAll(path.Dir(filepath), KubernetesDirPermissions); err != nil {
		return fmt.Errorf("failed to create directory %q [%v]", path.Dir(filepath), err)
	}

	// If err == nil, the file exists. Oops, we don't allow the file to exist already, fail.
	if _, err := os.Stat(filepath); err == nil {
		return fmt.Errorf("kubeconfig file %s already exists, but must not exist.", filepath)
	}

	if err := clientcmd.WriteToFile(*kubeconfig, filepath); err != nil {
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
