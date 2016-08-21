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

package app

import (
	"crypto/x509/pkix"
	"fmt"
	"io/ioutil"
	_ "net/http/pprof"
	"os"
	"path/filepath"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/certificates"
	unversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/fields"
	utilcertificates "k8s.io/kubernetes/pkg/util/certificates"
	"k8s.io/kubernetes/pkg/util/crypto"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	defaultKubeletClientCertificateFile = "kubelet-client.crt"
	defaultKubeletClientKeyFile         = "kubelet-client.key"
)

// bootstrapClientCert requests a client cert for kubelet if the kubeconfigPath file does not exist.
// The kubeconfig at bootstrapPath is used to request a client certificate from the API server.
// On success, a kubeconfig file referencing the generated key and obtained certificate is written to kubeconfigPath.
// The certificate and key file are stored in certDir.
func bootstrapClientCert(kubeconfigPath string, bootstrapPath string, certDir string, nodeName string) error {
	// Short-circuit if the kubeconfig file already exists.
	// TODO: inspect the kubeconfig, ensure a rest client can be built from it, verify client cert expiration, etc.
	_, err := os.Stat(kubeconfigPath)
	if err == nil {
		glog.V(2).Infof("Kubeconfig %s exists, skipping bootstrap", kubeconfigPath)
		return nil
	}
	if !os.IsNotExist(err) {
		glog.Errorf("Error reading kubeconfig %s, skipping bootstrap: %v", kubeconfigPath, err)
		return err
	}

	glog.V(2).Info("Using bootstrap kubeconfig to generate TLS client cert, key and kubeconfig file")

	bootstrapClientConfig, err := loadRESTClientConfig(bootstrapPath)
	if err != nil {
		return fmt.Errorf("unable to load bootstrap kubeconfig: %v", err)
	}
	bootstrapClient, err := unversionedcertificates.NewForConfig(bootstrapClientConfig)
	if err != nil {
		return fmt.Errorf("unable to create certificates signing request client: %v", err)
	}

	success := false

	// Get the private key.
	keyPath, err := filepath.Abs(filepath.Join(certDir, defaultKubeletClientKeyFile))
	if err != nil {
		return fmt.Errorf("unable to build bootstrap key path: %v", err)
	}
	keyData, generatedKeyFile, err := loadOrGenerateKeyFile(keyPath)
	if err != nil {
		return err
	}
	if generatedKeyFile {
		defer func() {
			if !success {
				if err := os.Remove(keyPath); err != nil {
					glog.Warningf("Cannot clean up the key file %q: %v", keyPath, err)
				}
			}
		}()
	}

	// Get the cert.
	certPath, err := filepath.Abs(filepath.Join(certDir, defaultKubeletClientCertificateFile))
	if err != nil {
		return fmt.Errorf("unable to build bootstrap client cert path: %v", err)
	}
	certData, err := RequestClientCertificate(bootstrapClient.CertificateSigningRequests(), keyData, nodeName)
	if err != nil {
		return err
	}
	if err := crypto.WriteCertToPath(certPath, certData); err != nil {
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

func loadOrGenerateKeyFile(keyPath string) (data []byte, wasGenerated bool, err error) {
	loadedData, err := ioutil.ReadFile(keyPath)
	if err == nil {
		return loadedData, false, err
	}
	if !os.IsNotExist(err) {
		return nil, false, fmt.Errorf("error loading key from %s: %v", keyPath, err)
	}

	generatedData, err := utilcertificates.GeneratePrivateKey()
	if err != nil {
		return nil, false, fmt.Errorf("error generating key: %v", err)
	}
	if err := crypto.WriteKeyToPath(keyPath, generatedData); err != nil {
		return nil, false, fmt.Errorf("error writing key to %s: %v", keyPath, err)
	}
	return generatedData, true, nil
}

// RequestClientCertificate will create a certificate signing request and send it to API server,
// then it will watch the object's status, once approved by API server, it will return the API
// server's issued certificate (pem-encoded). If there is any errors, or the watch timeouts,
// it will return an error.
func RequestClientCertificate(client unversionedcertificates.CertificateSigningRequestInterface, privateKeyData []byte, nodeName string) (certData []byte, err error) {
	subject := &pkix.Name{
		Organization: []string{"system:nodes"},
		CommonName:   fmt.Sprintf("system:node:%s", nodeName),
	}

	privateKey, err := utilcertificates.ParsePrivateKey(privateKeyData)
	if err != nil {
		return nil, fmt.Errorf("invalid private key for certificate request: %v", err)
	}
	csr, err := utilcertificates.NewCertificateRequest(privateKey, subject, nil, nil)
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
