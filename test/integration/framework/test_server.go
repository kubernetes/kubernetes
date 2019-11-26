/*
Copyright 2018 The Kubernetes Authors.

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

package framework

import (
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path"
	"testing"
	"time"

	"github.com/google/uuid"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	client "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/utils"
)

// TestServerSetup holds configuration information for a kube-apiserver test server.
type TestServerSetup struct {
	ModifyServerRunOptions func(*options.ServerRunOptions)
	ModifyServerConfig     func(*master.Config)
}

// StartTestServer runs a kube-apiserver, optionally calling out to the setup.ModifyServerRunOptions and setup.ModifyServerConfig functions
func StartTestServer(t *testing.T, stopCh <-chan struct{}, setup TestServerSetup) (client.Interface, *rest.Config) {
	certDir, _ := ioutil.TempDir("", "test-integration-"+t.Name())
	go func() {
		<-stopCh
		os.RemoveAll(certDir)
	}()

	_, defaultServiceClusterIPRange, _ := net.ParseCIDR("10.0.0.0/24")
	proxySigningKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	proxySigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "front-proxy-ca"}, proxySigningKey)
	if err != nil {
		t.Fatal(err)
	}
	proxyCACertFile, _ := ioutil.TempFile(certDir, "proxy-ca.crt")
	if err := ioutil.WriteFile(proxyCACertFile.Name(), utils.EncodeCertPEM(proxySigningCert), 0644); err != nil {
		t.Fatal(err)
	}
	clientSigningKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	clientSigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, clientSigningKey)
	if err != nil {
		t.Fatal(err)
	}
	clientCACertFile, _ := ioutil.TempFile(certDir, "client-ca.crt")
	if err := ioutil.WriteFile(clientCACertFile.Name(), utils.EncodeCertPEM(clientSigningCert), 0644); err != nil {
		t.Fatal(err)
	}

	listener, _, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	kubeAPIServerOptions := options.NewServerRunOptions()
	kubeAPIServerOptions.SecureServing.Listener = listener
	kubeAPIServerOptions.SecureServing.BindAddress = net.ParseIP("127.0.0.1")
	kubeAPIServerOptions.SecureServing.ServerCert.CertDirectory = certDir
	kubeAPIServerOptions.InsecureServing.BindPort = 0
	kubeAPIServerOptions.Etcd.StorageConfig.Prefix = path.Join("/", uuid.New().String(), "registry")
	kubeAPIServerOptions.Etcd.StorageConfig.Transport.ServerList = []string{GetEtcdURL()}
	kubeAPIServerOptions.ServiceClusterIPRanges = defaultServiceClusterIPRange.String()
	kubeAPIServerOptions.Authentication.RequestHeader.UsernameHeaders = []string{"X-Remote-User"}
	kubeAPIServerOptions.Authentication.RequestHeader.GroupHeaders = []string{"X-Remote-Group"}
	kubeAPIServerOptions.Authentication.RequestHeader.ExtraHeaderPrefixes = []string{"X-Remote-Extra-"}
	kubeAPIServerOptions.Authentication.RequestHeader.AllowedNames = []string{"kube-aggregator"}
	kubeAPIServerOptions.Authentication.RequestHeader.ClientCAFile = proxyCACertFile.Name()
	kubeAPIServerOptions.Authentication.ClientCert.ClientCA = clientCACertFile.Name()
	kubeAPIServerOptions.Authorization.Modes = []string{"Node", "RBAC"}

	if setup.ModifyServerRunOptions != nil {
		setup.ModifyServerRunOptions(kubeAPIServerOptions)
	}

	completedOptions, err := app.Complete(kubeAPIServerOptions)
	if err != nil {
		t.Fatal(err)
	}
	tunneler, proxyTransport, err := app.CreateNodeDialer(completedOptions)
	if err != nil {
		t.Fatal(err)
	}
	kubeAPIServerConfig, _, _, _, err := app.CreateKubeAPIServerConfig(completedOptions, tunneler, proxyTransport)
	if err != nil {
		t.Fatal(err)
	}

	if setup.ModifyServerConfig != nil {
		setup.ModifyServerConfig(kubeAPIServerConfig)
	}
	kubeAPIServer, err := app.CreateKubeAPIServer(kubeAPIServerConfig, genericapiserver.NewEmptyDelegate())
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		if err := kubeAPIServer.GenericAPIServer.PrepareRun().Run(stopCh); err != nil {
			t.Fatal(err)
		}
	}()

	// Adjust the loopback config for external use (external server name and CA)
	kubeAPIServerClientConfig := rest.CopyConfig(kubeAPIServerConfig.GenericConfig.LoopbackClientConfig)
	kubeAPIServerClientConfig.CAFile = path.Join(certDir, "apiserver.crt")
	kubeAPIServerClientConfig.CAData = nil
	kubeAPIServerClientConfig.ServerName = ""

	// wait for health
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		healthzConfig := rest.CopyConfig(kubeAPIServerClientConfig)
		healthzConfig.ContentType = ""
		healthzConfig.AcceptContentTypes = ""
		kubeClient, err := client.NewForConfig(healthzConfig)
		if err != nil {
			// this happens because we race the API server start
			t.Log(err)
			return false, nil
		}

		healthStatus := 0
		kubeClient.Discovery().RESTClient().Get().AbsPath("/healthz").Do().StatusCode(&healthStatus)
		if healthStatus != http.StatusOK {
			return false, nil
		}

		if _, err := kubeClient.CoreV1().Namespaces().Get("default", metav1.GetOptions{}); err != nil {
			return false, nil
		}
		if _, err := kubeClient.CoreV1().Namespaces().Get("kube-system", metav1.GetOptions{}); err != nil {
			return false, nil
		}

		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	kubeAPIServerClient, err := client.NewForConfig(kubeAPIServerClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	return kubeAPIServerClient, kubeAPIServerClientConfig
}
