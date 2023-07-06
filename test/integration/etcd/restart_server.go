/*
Copyright 2023 The Kubernetes Authors.

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

/*
The etcd package can be utilized in tests that involve the handling of
ETCD storage data. It starts an API server that can be used for tests that require
one of every type of resource. If a test requires the restart of an API server,
then StartRealAPIServerOrDieForKMS and Restart can be used. It can restart the API server
without impacting the ETCD data or affecting client connections.
*/
package etcd

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/pflag"
	clientv3 "go.etcd.io/etcd/client/v3"

	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	_ "k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
	netutils "k8s.io/utils/net"
)

func createLocalhostListenerOnFreePort(t *testing.T, port int, errCh chan error) (net.Listener, int, error) {
	address := fmt.Sprintf("127.0.0.1:%d", port)
	var listener net.Listener
	if err := wait.PollImmediate(time.Second, 2*time.Minute, func() (done bool, err error) {
		select {
		case err := <-errCh:
			t.Logf("port check errCh err %v", err)
			return false, err
		default:
		}
		// wait for the port to be free
		listener, err = net.Listen("tcp", address)
		if err == nil {
			return true, nil
		}
		if strings.Contains(err.Error(), "address already in use") {
			t.Logf("Address %s is in use. Waiting...\n", address)
		} else {
			t.Log(err)
		}
		return false, nil
	}); err != nil {
		t.Log(err)
	}

	if port == 0 {
		tcpAddr, ok := listener.Addr().(*net.TCPAddr)
		if !ok {
			listener.Close()
			return nil, 0, fmt.Errorf("invalid listen address: %q", listener.Addr().String())
		}
		port = tcpAddr.Port
	}

	return listener, port, nil
}
func writeCACertFiles(t *testing.T, certDir string) ([]byte, string, *x509.Certificate, *rsa.PrivateKey) {
	caKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	caCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, caKey)
	if err != nil {
		t.Fatal(err)
	}
	caCertPEM := utils.EncodeCertPEM(caCert)

	caFilename := filepath.Join(certDir, "ca.crt")

	if err := os.WriteFile(caFilename, utils.EncodeCertPEM(caCert), 0644); err != nil {
		t.Fatal(err)
	}

	return caCertPEM, caFilename, caCert, caKey
}

func writeClientCerts(t *testing.T, caCert *x509.Certificate, caKey *rsa.PrivateKey, certDir string, duration time.Duration) ([]byte, []byte) {
	key, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	keyBytes, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	keyPEM := new(bytes.Buffer)
	err = pem.Encode(keyPEM, &pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: keyBytes,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(certDir, "client.key"), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes}), 0666); err != nil {
		t.Fatal(err)
	}

	// returns a uniform random value in [0, max-1), then add 1 to serial to make it a uniform random value in [1, max).
	serial, err := rand.Int(rand.Reader, new(big.Int).SetInt64(math.MaxInt64-1))
	if err != nil {
		t.Fatal(err)
	}
	serial = new(big.Int).Add(serial, big.NewInt(1))

	certTmpl := x509.Certificate{
		Subject: pkix.Name{
			CommonName:   "foo",
			Organization: []string{"system:masters"},
		},
		SerialNumber: serial,
		NotBefore:    caCert.NotBefore,
		NotAfter:     time.Now().Add(duration).UTC(),
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	certDERBytes, err := x509.CreateCertificate(rand.Reader, &certTmpl, caCert, key.Public(), caKey)
	if err != nil {
		t.Fatal(err)
	}

	certPEM := new(bytes.Buffer)
	err = pem.Encode(certPEM, &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: certDERBytes,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(certDir, "client.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDERBytes}), 0666); err != nil {
		t.Fatal(err)
	}

	return certPEM.Bytes(), keyPEM.Bytes()
}

func writeServingCerts(t *testing.T, caCert *x509.Certificate, caKey *rsa.PrivateKey, certDir string, duration time.Duration) (string, string) {
	key, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	keyBytes, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}

	keyPEM := new(bytes.Buffer)
	err = pem.Encode(keyPEM, &pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: keyBytes,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(certDir, "serving.key"), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes}), 0666); err != nil {
		t.Fatal(err)
	}

	// returns a uniform random value in [0, max-1), then add 1 to serial to make it a uniform random value in [1, max).
	serial, err := rand.Int(rand.Reader, new(big.Int).SetInt64(math.MaxInt64-1))
	if err != nil {
		t.Fatal(err)
	}
	serial = new(big.Int).Add(serial, big.NewInt(1))

	certTmpl := x509.Certificate{
		Subject: pkix.Name{
			CommonName: "foo",
		},
		DNSNames:     []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes"},
		SerialNumber: serial,
		NotBefore:    caCert.NotBefore,
		NotAfter:     time.Now().Add(duration).UTC(),
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}

	certDERBytes, err := x509.CreateCertificate(rand.Reader, &certTmpl, caCert, key.Public(), caKey)
	if err != nil {
		t.Fatal(err)
	}
	certPEM := new(bytes.Buffer)
	err = pem.Encode(certPEM, &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: certDERBytes,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(certDir, "serving.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDERBytes}), 0666); err != nil {
		t.Fatal(err)
	}

	return filepath.Join(certDir, "serving.crt"), filepath.Join(certDir, "serving.key")
}

// StartRealAPIServerOrDieForKMS starts an API server that is appropriate for use in tests that require one of every resource
// and it creates all the test CRDs
func StartRealAPIServerOrDieForKMS(t *testing.T, customFlags []string) (*APIServer, error) {
	certDir := t.TempDir()
	_, defaultServiceClusterIPRange, err := netutils.ParseCIDRSloppy("10.0.0.0/24")
	if err != nil {
		t.Fatal(err)
	}

	saSigningKeyFile, err := os.CreateTemp("/tmp", "insecure_test_key")
	if err != nil {
		t.Fatalf("create temp file failed: %v", err)
	}
	if err = os.WriteFile(saSigningKeyFile.Name(), []byte(ecdsaPrivateKey), 0666); err != nil {
		t.Fatalf("write file %s failed: %v", saSigningKeyFile.Name(), err)
	}

	kubeAPIServerOptions := options.NewServerRunOptions()
	kubeAPIServerOptions.SecureServing.BindPort = 0
	kubeAPIServerOptions.ServiceAccountSigningKeyFile = saSigningKeyFile.Name()
	kubeAPIServerOptions.Etcd.StorageConfig.Transport.ServerList = []string{framework.GetEtcdURL()}
	kubeAPIServerOptions.ServiceClusterIPRanges = defaultServiceClusterIPRange.String()
	kubeAPIServerOptions.Authentication.ServiceAccounts.Issuers = []string{"https://foo.bar.example.com"}
	kubeAPIServerOptions.Authentication.ServiceAccounts.KeyFiles = []string{saSigningKeyFile.Name()}
	kubeAPIServerOptions.APIEnablement.RuntimeConfig["api/all"] = "true"
	// we make requests to all resources, don't log warnings about deprecated ones
	restclient.SetDefaultWarningHandler(restclient.NoWarnings{})
	caData, caFilename, caCert, caKey := writeCACertFiles(t, certDir)
	clientCertData, clientKeyData := writeClientCerts(t, caCert, caKey, certDir, 10_000*time.Hour)
	servingCertFilename, servingKeyFilename := writeServingCerts(t, caCert, caKey, certDir, 10_000*time.Hour)
	// do not need to set CertDirectory since we are providing our own certs
	kubeAPIServerOptions.Authentication.ClientCert.ClientCA = caFilename
	kubeAPIServerOptions.SecureServing.ServerCert.CertKey.KeyFile = servingKeyFilename
	kubeAPIServerOptions.SecureServing.ServerCert.CertKey.CertFile = servingCertFilename

	rawClient, kvClient, err := integration.GetEtcdClients(kubeAPIServerOptions.Etcd.StorageConfig.Transport)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { rawClient.Close() })
	// make sure we start with a clean slate
	if _, err := kvClient.Delete(context.Background(), "/registry/", clientv3.WithPrefix()); err != nil {
		t.Fatal(err)
	}

	stopCh := make(chan struct{})
	errCh := make(chan error, 100)

	cleanup := start(t, stopCh, errCh, kubeAPIServerOptions, nil, customFlags, saSigningKeyFile, certDir, rawClient)

	// need to do this AFTER start() to get the port
	address := fmt.Sprintf("127.0.0.1:%d", kubeAPIServerOptions.SecureServing.BindPort)

	kubeClientConfig := &restclient.Config{
		Host: address,
		TLSClientConfig: restclient.TLSClientConfig{
			ServerName: "kubernetes.default.svc",
			CAData:     caData,
			CertData:   clientCertData,
			KeyData:    clientKeyData,
		},
		QPS:   99999,
		Burst: 9999,
	}

	kubeClient := clientset.NewForConfigOrDie(kubeClientConfig)
	klog.Infof("kubeClientConfig.Host: %v", kubeClientConfig.Host)

	err = healthCheck(t, kubeClient, errCh)
	if err != nil {
		// need to clean up here instead of test.cleanup since apiServer has not been created yet
		if rawClient != nil {
			rawClient.Close()
		}
		cleanup()
		return nil, fmt.Errorf("healthCheck failed with err: %v", err)
	}

	// create CRDs so we can make sure that custom resources do not get lost
	CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(kubeClientConfig), false, GetCustomResourceDefinitionData()...)

	return &APIServer{
		Client:           kubeClient,
		Dynamic:          dynamic.NewForConfigOrDie(kubeClientConfig),
		Config:           kubeClientConfig,
		KV:               kvClient,
		Cleanup:          cleanup,
		ServerOpts:       kubeAPIServerOptions,
		EtcdClient:       rawClient,
		StopCh:           stopCh,
		ErrCh:            errCh,
		SaSigningKeyFile: saSigningKeyFile,
		CertDir:          certDir,
	}, nil
}

func healthCheck(t *testing.T, kubeClient *clientset.Clientset, errCh chan error) error {
	lastHealth := ""
	attempt := 0
	if err := wait.PollImmediate(time.Second, time.Minute, func() (done bool, err error) {
		select {
		case err := <-errCh:
			return false, err
		default:
		}

		// wait for the server to be healthy
		result := kubeClient.RESTClient().Get().AbsPath("/healthz").Do(context.TODO())
		content, _ := result.Raw()
		lastHealth = string(content)
		if errResult := result.Error(); errResult != nil {
			attempt++
			if attempt < 10 {
				t.Log("waiting for server to be healthy")
			} else {
				t.Log(errResult)
			}
			return false, nil
		}
		var status int
		result.StatusCode(&status)
		return status == http.StatusOK, nil
	}); err != nil {
		t.Log(lastHealth)
		return err
	}
	return nil
}

// Restart restarts an API server
func Restart(t *testing.T, customFlags []string, oldServer *APIServer) error {
	klog.Infof("Restart() customFlags: %v", customFlags)
	close(oldServer.StopCh)
	stopCh := make(chan struct{})
	errCh := make(chan error, 100)

	oldServer.Cleanup = nil
	cleanup := start(t, stopCh, errCh, oldServer.ServerOpts, &oldServer.ServerOpts.Etcd.StorageConfig, customFlags, oldServer.SaSigningKeyFile, oldServer.CertDir, oldServer.EtcdClient)

	oldServer.StopCh = stopCh
	oldServer.ErrCh = errCh
	oldServer.Cleanup = cleanup

	err := healthCheck(t, oldServer.Client.(*clientset.Clientset), errCh)
	if err != nil {
		// let test.cleanup do the cleanup
		return fmt.Errorf("healthCheck failed with err: %v", err)
	}
	return err

}
func start(t *testing.T, stopCh chan struct{}, errCh chan error, kubeAPIServerOptions *options.ServerRunOptions, storageConfig *storagebackend.Config, customFlags []string, saSigningKeyFile *os.File, certDir string, etcdClient *clientv3.Client) (cleanupFunc func()) {
	cleanup := func() {
		close(stopCh)
		if errCh != nil {
			err, ok := <-errCh
			if ok && err != nil {
				klog.Errorf("Failed to shutdown test server clearly: %v", err)
			}
		}
		utiltesting.CloseAndRemove(t, saSigningKeyFile)
		if err := os.RemoveAll(certDir); err != nil {
			t.Log(err)
		}
	}
	defer func() {
		// failed before we can set APIServer.Cleanup function
		if cleanupFunc == nil {
			close(errCh)
			if etcdClient != nil {
				etcdClient.Close()
			}
			cleanup()
		}
	}()

	if storageConfig == nil {
		kubeAPIServerOptions.Etcd.StorageConfig = *framework.SharedEtcd()
	} else {
		kubeAPIServerOptions.Etcd.StorageConfig = *storageConfig
	}

	listener, port, err := createLocalhostListenerOnFreePort(t, kubeAPIServerOptions.SecureServing.BindPort, errCh)
	if err != nil {
		t.Fatalf("could not create listener: %v", err)
	}
	kubeAPIServerOptions.SecureServing.Listener = listener
	kubeAPIServerOptions.SecureServing.BindPort = port

	fs := pflag.NewFlagSet("test", pflag.PanicOnError)
	for _, f := range kubeAPIServerOptions.Flags().FlagSets {
		fs.AddFlagSet(f)
	}
	if err := fs.Parse(customFlags); err != nil {
		t.Fatal(err)
	}
	completedOptions, err := kubeAPIServerOptions.Complete()
	if err != nil {
		t.Fatal(err)
	}

	if errs := completedOptions.Validate(); len(errs) != 0 {
		t.Fatalf("failed to validate ServerRunOptions: %v", utilerrors.NewAggregate(errs))
	}

	config, err := app.NewConfig(completedOptions)
	if err != nil {
		t.Fatal(err)
	}
	completed, err := config.Complete()
	if err != nil {
		t.Fatal(err)
	}
	kubeAPIServer, err := app.CreateServerChain(completed)
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		defer close(errCh)
		// Catch panics that occur in this go routine so we get a comprehensible failure
		defer func() {
			if err := recover(); err != nil {
				errCh <- fmt.Errorf("err from recover: %#v", err)
				return
			}
		}()
		prepared, err := kubeAPIServer.PrepareRun()
		if err != nil {
			errCh <- err
			return
		}

		if err := prepared.Run(stopCh); err != nil {
			errCh <- err
			return
		}
	}()
	cleanupFunc = cleanup
	return cleanupFunc
}
