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

package tls

import (
	"crypto/tls"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	client "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

func runBasicSecureAPIServer(t *testing.T, ciphers []string) (uint32, error) {
	certDir, _ := ioutil.TempDir("", "test-integration-tls")
	defer os.RemoveAll(certDir)
	_, defaultServiceClusterIPRange, _ := net.ParseCIDR("10.0.0.0/24")
	kubeClientConfigValue := atomic.Value{}
	var kubePort uint32

	go func() {
		listener, port, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}

		atomic.StoreUint32(&kubePort, uint32(port))

		kubeAPIServerOptions := options.NewServerRunOptions()
		kubeAPIServerOptions.SecureServing.BindAddress = net.ParseIP("127.0.0.1")
		kubeAPIServerOptions.SecureServing.BindPort = port
		kubeAPIServerOptions.SecureServing.Listener = listener
		kubeAPIServerOptions.SecureServing.ServerCert.CertDirectory = certDir
		kubeAPIServerOptions.SecureServing.CipherSuites = ciphers
		kubeAPIServerOptions.InsecureServing.BindPort = 0
		kubeAPIServerOptions.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURL()}
		kubeAPIServerOptions.ServiceClusterIPRange = *defaultServiceClusterIPRange

		tunneler, proxyTransport, err := app.CreateNodeDialer(kubeAPIServerOptions)
		if err != nil {
			t.Fatal(err)
		}
		kubeAPIServerConfig, sharedInformers, versionedInformers, _, _, _, err := app.CreateKubeAPIServerConfig(kubeAPIServerOptions, tunneler, proxyTransport)
		if err != nil {
			t.Fatal(err)
		}
		kubeAPIServerConfig.ExtraConfig.EnableCoreControllers = false
		kubeClientConfigValue.Store(kubeAPIServerConfig.GenericConfig.LoopbackClientConfig)

		kubeAPIServer, err := app.CreateKubeAPIServer(kubeAPIServerConfig, genericapiserver.EmptyDelegate, sharedInformers, versionedInformers)
		if err != nil {
			t.Fatal(err)
		}

		if err := kubeAPIServer.GenericAPIServer.PrepareRun().Run(wait.NeverStop); err != nil {
			t.Log(err)
		}
		time.Sleep(100 * time.Millisecond)
	}()

	// Ensure server is ready
	err := wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		obj := kubeClientConfigValue.Load()
		if obj == nil {
			return false, nil
		}
		kubeClientConfig := kubeClientConfigValue.Load().(*rest.Config)
		kubeClientConfig.ContentType = ""
		kubeClientConfig.AcceptContentTypes = ""
		kubeClient, err := client.NewForConfig(kubeClientConfig)
		if err != nil {
			// this happens because we race the API server start
			t.Log(err)
			return false, nil
		}
		if _, err := kubeClient.Discovery().ServerVersion(); err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return 0, err
	}

	securePort := atomic.LoadUint32(&kubePort)
	return securePort, nil
}

func TestAPICiphers(t *testing.T) {

	basicServerCiphers := []string{"TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305", "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305", "TLS_RSA_WITH_AES_128_CBC_SHA", "TLS_RSA_WITH_AES_256_CBC_SHA", "TLS_RSA_WITH_AES_128_GCM_SHA256", "TLS_RSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA"}

	kubePort, err := runBasicSecureAPIServer(t, basicServerCiphers)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		clientCiphers []uint16
		expectedError bool
	}{
		{
			// Not supported cipher
			clientCiphers: []uint16{tls.TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA},
			expectedError: true,
		},
		{
			// Supported cipher
			clientCiphers: []uint16{tls.TLS_RSA_WITH_AES_256_CBC_SHA},
			expectedError: false,
		},
	}

	for i, test := range tests {
		runTestAPICiphers(t, i, kubePort, test.clientCiphers, test.expectedError)
	}
}

func runTestAPICiphers(t *testing.T, testID int, kubePort uint32, clientCiphers []uint16, expectedError bool) {

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
			CipherSuites:       clientCiphers,
		},
	}
	client := &http.Client{Transport: tr}
	req, err := http.NewRequest("GET", fmt.Sprintf("https://127.0.0.1:%d", kubePort), nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Do(req)

	if expectedError == true && err == nil {
		t.Fatalf("%d: expecting error for cipher test, client cipher is supported and it should't", testID)
	} else if err != nil && expectedError == false {
		t.Fatalf("%d: not expecting error by client with cipher failed: %+v", testID, err)
	}

	if err == nil {
		defer resp.Body.Close()
	}
}
