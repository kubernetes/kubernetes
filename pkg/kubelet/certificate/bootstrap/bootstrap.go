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

package bootstrap

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	certificates "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/transport"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/kubelet/util/csr"
)

const (
	defaultKubeletClientCertificateFile = "kubelet-client.crt"
	defaultKubeletClientKeyFile         = "kubelet-client.key"
)

// LoadClientCert requests a client cert for kubelet if the kubeconfigPath file does not exist.
// The kubeconfig at bootstrapPath is used to request a client certificate from the API server.
// On success, a kubeconfig file referencing the generated key and obtained certificate is written to kubeconfigPath.
// The certificate and key file are stored in certDir.
func LoadClientCert(kubeconfigPath string, bootstrapPath string, certDir string, nodeName types.NodeName) error {
	// Short-circuit if the kubeconfig file exists and is valid.
	ok, err := verifyBootstrapClientConfig(kubeconfigPath)
	if err != nil {
		return err
	}
	if ok {
		glog.V(2).Infof("Kubeconfig %s exists and is valid, skipping bootstrap", kubeconfigPath)
		return nil
	}

	glog.V(2).Info("Using bootstrap kubeconfig to generate TLS client cert, key and kubeconfig file")

	bootstrapClientConfig, err := loadRESTClientConfig(bootstrapPath)
	if err != nil {
		return fmt.Errorf("unable to load bootstrap kubeconfig: %v", err)
	}
	bootstrapClient, err := certificates.NewForConfig(bootstrapClientConfig)
	if err != nil {
		return fmt.Errorf("unable to create certificates signing request client: %v", err)
	}

	success := false

	// Get the private key.
	keyPath, err := filepath.Abs(filepath.Join(certDir, defaultKubeletClientKeyFile))
	if err != nil {
		return fmt.Errorf("unable to build bootstrap key path: %v", err)
	}
	keyData, _, err := certutil.LoadOrGenerateKeyFile(keyPath)
	if err != nil {
		return err
	}

	// Get the cert.
	certPath, err := filepath.Abs(filepath.Join(certDir, defaultKubeletClientCertificateFile))
	if err != nil {
		return fmt.Errorf("unable to build bootstrap client cert path: %v", err)
	}
	certData, err := csr.RequestNodeCertificate(bootstrapClient.CertificateSigningRequests(), keyData, nodeName)
	if err != nil {
		return err
	}
	if err := certutil.WriteCert(certPath, certData); err != nil {
		return err
	}
	defer func() {
		if !success {
			if err := os.Remove(certPath); err != nil {
				glog.Warningf("Cannot clean up the cert file %q: %v", certPath, err)
			}
		}
	}()

	// Get the CA data from the bootstrap client config.
	caFile, caData := bootstrapClientConfig.CAFile, []byte{}
	if len(caFile) == 0 {
		caData = bootstrapClientConfig.CAData
	}

	// Build resulting kubeconfig.
	kubeconfigData := clientcmdapi.Config{
		// Define a cluster stanza based on the bootstrap kubeconfig.
		Clusters: map[string]*clientcmdapi.Cluster{"default-cluster": {
			Server:                   bootstrapClientConfig.Host,
			InsecureSkipTLSVerify:    bootstrapClientConfig.Insecure,
			CertificateAuthority:     caFile,
			CertificateAuthorityData: caData,
		}},
		// Define auth based on the obtained client cert.
		AuthInfos: map[string]*clientcmdapi.AuthInfo{"default-auth": {
			ClientCertificate: certPath,
			ClientKey:         keyPath,
		}},
		// Define a context that connects the auth info and cluster, and set it as the default
		Contexts: map[string]*clientcmdapi.Context{"default-context": {
			Cluster:   "default-cluster",
			AuthInfo:  "default-auth",
			Namespace: "default",
		}},
		CurrentContext: "default-context",
	}

	// Marshal to disk
	if err := clientcmd.WriteToFile(kubeconfigData, kubeconfigPath); err != nil {
		return err
	}

	success = true
	return nil
}

func loadRESTClientConfig(kubeconfig string) (*restclient.Config, error) {
	// Load structured kubeconfig data from the given path.
	loader := &clientcmd.ClientConfigLoadingRules{ExplicitPath: kubeconfig}
	loadedConfig, err := loader.Load()
	if err != nil {
		return nil, err
	}
	// Flatten the loaded data to a particular restclient.Config based on the current context.
	return clientcmd.NewNonInteractiveClientConfig(
		*loadedConfig,
		loadedConfig.CurrentContext,
		&clientcmd.ConfigOverrides{},
		loader,
	).ClientConfig()
}

// verifyBootstrapClientConfig checks the provided kubeconfig to see if it has a valid
// client certificate. It returns true if the kubeconfig is valid, or an error if bootstrapping
// should stop immediately.
func verifyBootstrapClientConfig(kubeconfigPath string) (bool, error) {
	_, err := os.Stat(kubeconfigPath)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("error reading existing bootstrap kubeconfig %s: %v", kubeconfigPath, err)
	}
	bootstrapClientConfig, err := loadRESTClientConfig(kubeconfigPath)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to read existing bootstrap client config: %v", err))
		return false, nil
	}
	transportConfig, err := bootstrapClientConfig.TransportConfig()
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to load transport configuration from existing bootstrap client config: %v", err))
		return false, nil
	}
	// has side effect of populating transport config data fields
	if _, err := transport.TLSConfigFor(transportConfig); err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to load TLS configuration from existing bootstrap client config: %v", err))
		return false, nil
	}
	certs, err := certutil.ParseCertsPEM(transportConfig.TLS.CertData)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to load TLS certificates from existing bootstrap client config: %v", err))
		return false, nil
	}
	if len(certs) == 0 {
		utilruntime.HandleError(fmt.Errorf("Unable to read TLS certificates from existing bootstrap client config: %v", err))
		return false, nil
	}
	now := time.Now()
	for _, cert := range certs {
		if now.After(cert.NotAfter) {
			utilruntime.HandleError(fmt.Errorf("Part of the existing bootstrap client certificate is expired: %s", cert.NotAfter))
			return false, nil
		}
	}
	return true, nil
}
