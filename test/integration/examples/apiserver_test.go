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

package apiserver

import (
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	client "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/cert"
	apiregistrationv1beta1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	kubeaggregatorserver "k8s.io/kube-aggregator/pkg/cmd/server"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	sampleserver "k8s.io/sample-apiserver/pkg/cmd/server"
)

var groupVersion = v1alpha1.SchemeGroupVersion

var groupVersionForDiscovery = metav1.GroupVersionForDiscovery{
	GroupVersion: groupVersion.String(),
	Version:      groupVersion.Version,
}

func TestAggregatedAPIServer(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	certDir, _ := ioutil.TempDir("", "test-integration-apiserver")
	defer os.RemoveAll(certDir)
	_, defaultServiceClusterIPRange, _ := net.ParseCIDR("10.0.0.0/24")
	proxySigningKey, err := cert.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	proxySigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "front-proxy-ca"}, proxySigningKey)
	if err != nil {
		t.Fatal(err)
	}
	proxyCACertFile, _ := ioutil.TempFile(certDir, "proxy-ca.crt")
	if err := ioutil.WriteFile(proxyCACertFile.Name(), cert.EncodeCertPEM(proxySigningCert), 0644); err != nil {
		t.Fatal(err)
	}
	clientSigningKey, err := cert.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	clientSigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, clientSigningKey)
	if err != nil {
		t.Fatal(err)
	}
	clientCACertFile, _ := ioutil.TempFile(certDir, "client-ca.crt")
	if err := ioutil.WriteFile(clientCACertFile.Name(), cert.EncodeCertPEM(clientSigningCert), 0644); err != nil {
		t.Fatal(err)
	}

	kubeClientConfigValue := atomic.Value{}
	go func() {
		for {
			// always get a fresh port in case something claimed the old one
			kubePort, err := framework.FindFreeLocalPort()
			if err != nil {
				t.Fatal(err)
			}

			kubeAPIServerOptions := options.NewServerRunOptions()
			kubeAPIServerOptions.SecureServing.BindAddress = net.ParseIP("127.0.0.1")
			kubeAPIServerOptions.SecureServing.BindPort = kubePort
			kubeAPIServerOptions.SecureServing.ServerCert.CertDirectory = certDir
			kubeAPIServerOptions.InsecureServing.BindPort = 0
			kubeAPIServerOptions.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURLFromEnv()}
			kubeAPIServerOptions.ServiceClusterIPRange = *defaultServiceClusterIPRange
			kubeAPIServerOptions.Authentication.RequestHeader.UsernameHeaders = []string{"X-Remote-User"}
			kubeAPIServerOptions.Authentication.RequestHeader.GroupHeaders = []string{"X-Remote-Group"}
			kubeAPIServerOptions.Authentication.RequestHeader.ExtraHeaderPrefixes = []string{"X-Remote-Extra-"}
			kubeAPIServerOptions.Authentication.RequestHeader.AllowedNames = []string{"kube-aggregator"}
			kubeAPIServerOptions.Authentication.RequestHeader.ClientCAFile = proxyCACertFile.Name()
			kubeAPIServerOptions.Authentication.ClientCert.ClientCA = clientCACertFile.Name()
			kubeAPIServerOptions.Authorization.Mode = "RBAC"

			tunneler, proxyTransport, err := app.CreateNodeDialer(kubeAPIServerOptions)
			if err != nil {
				t.Fatal(err)
			}
			kubeAPIServerConfig, sharedInformers, _, _, _, err := app.CreateKubeAPIServerConfig(kubeAPIServerOptions, tunneler, proxyTransport)
			if err != nil {
				t.Fatal(err)
			}
			kubeClientConfigValue.Store(kubeAPIServerConfig.GenericConfig.LoopbackClientConfig)

			kubeAPIServer, err := app.CreateKubeAPIServer(kubeAPIServerConfig, genericapiserver.EmptyDelegate, sharedInformers, nil)
			if err != nil {
				t.Fatal(err)
			}

			if err := kubeAPIServer.GenericAPIServer.PrepareRun().Run(wait.NeverStop); err != nil {
				t.Log(err)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()

	// just use json because everyone speaks it
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (done bool, err error) {
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
		t.Fatal(err)
	}

	// after this point we won't be mutating, so the race detector will be fine
	kubeClientConfig := kubeClientConfigValue.Load().(*rest.Config)

	// write a kubeconfig out for starting other API servers with delegated auth.  remember, no in-cluster config
	adminKubeConfig := createKubeConfig(kubeClientConfig)
	kubeconfigFile, _ := ioutil.TempFile("", "")
	defer os.Remove(kubeconfigFile.Name())
	clientcmd.WriteToFile(*adminKubeConfig, kubeconfigFile.Name())
	wardleCertDir, _ := ioutil.TempDir("", "test-integration-wardle-server")
	defer os.RemoveAll(wardleCertDir)
	wardlePort := new(int32)

	// start the wardle server to prove we can aggregate it
	go func() {
		for {
			// always get a fresh port in case something claimed the old one
			wardlePortInt, err := framework.FindFreeLocalPort()
			if err != nil {
				t.Fatal(err)
			}
			atomic.StoreInt32(wardlePort, int32(wardlePortInt))
			wardleCmd := sampleserver.NewCommandStartWardleServer(os.Stdout, os.Stderr, stopCh)
			wardleCmd.SetArgs([]string{
				"--bind-address", "127.0.0.1",
				"--secure-port", strconv.Itoa(wardlePortInt),
				"--requestheader-username-headers=X-Remote-User",
				"--requestheader-group-headers=X-Remote-Group",
				"--requestheader-extra-headers-prefix=X-Remote-Extra-",
				"--requestheader-client-ca-file=" + proxyCACertFile.Name(),
				"--requestheader-allowed-names=kube-aggregator",
				"--authentication-kubeconfig", kubeconfigFile.Name(),
				"--authorization-kubeconfig", kubeconfigFile.Name(),
				"--etcd-servers", framework.GetEtcdURLFromEnv(),
				"--cert-dir", wardleCertDir,
			})
			if err := wardleCmd.Execute(); err != nil {
				t.Log(err)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()

	wardleClientConfig := rest.AnonymousClientConfig(kubeClientConfig)
	wardleClientConfig.CAFile = path.Join(wardleCertDir, "apiserver.crt")
	wardleClientConfig.CAData = nil
	wardleClientConfig.ServerName = ""
	wardleClientConfig.BearerToken = kubeClientConfig.BearerToken
	var wardleClient client.Interface
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		wardleClientConfig.Host = fmt.Sprintf("https://127.0.0.1:%d", atomic.LoadInt32(wardlePort))
		wardleClient, err = client.NewForConfig(wardleClientConfig)
		if err != nil {
			// this happens because we race the API server start
			t.Log(err)
			return false, nil
		}
		if _, err := wardleClient.Discovery().ServerVersion(); err != nil {
			t.Log(err)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// start the aggregator
	aggregatorCertDir, _ := ioutil.TempDir("", "test-integration-aggregator")
	defer os.RemoveAll(aggregatorCertDir)
	proxyClientKey, err := cert.NewPrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	proxyClientCert, err := cert.NewSignedCert(
		cert.Config{
			CommonName: "kube-aggregator",
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		},
		proxyClientKey, proxySigningCert, proxySigningKey,
	)
	proxyClientCertFile, _ := ioutil.TempFile(aggregatorCertDir, "proxy-client.crt")
	proxyClientKeyFile, _ := ioutil.TempFile(aggregatorCertDir, "proxy-client.key")
	if err := ioutil.WriteFile(proxyClientCertFile.Name(), cert.EncodeCertPEM(proxyClientCert), 0600); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(proxyClientKeyFile.Name(), cert.EncodePrivateKeyPEM(proxyClientKey), 0644); err != nil {
		t.Fatal(err)
	}
	aggregatorPort := new(int32)

	go func() {
		for {
			// always get a fresh port in case something claimed the old one
			aggregatorPortInt, err := framework.FindFreeLocalPort()
			if err != nil {
				t.Fatal(err)
			}
			atomic.StoreInt32(aggregatorPort, int32(aggregatorPortInt))
			aggregatorCmd := kubeaggregatorserver.NewCommandStartAggregator(os.Stdout, os.Stderr, stopCh)
			aggregatorCmd.SetArgs([]string{
				"--bind-address", "127.0.0.1",
				"--secure-port", strconv.Itoa(aggregatorPortInt),
				"--requestheader-username-headers", "",
				"--proxy-client-cert-file", proxyClientCertFile.Name(),
				"--proxy-client-key-file", proxyClientKeyFile.Name(),
				"--core-kubeconfig", kubeconfigFile.Name(),
				"--authentication-kubeconfig", kubeconfigFile.Name(),
				"--authorization-kubeconfig", kubeconfigFile.Name(),
				"--etcd-servers", framework.GetEtcdURLFromEnv(),
				"--cert-dir", aggregatorCertDir,
			})
			if err := aggregatorCmd.Execute(); err != nil {
				t.Log(err)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()

	aggregatorClientConfig := rest.AnonymousClientConfig(kubeClientConfig)
	aggregatorClientConfig.CAFile = path.Join(aggregatorCertDir, "apiserver.crt")
	aggregatorClientConfig.CAData = nil
	aggregatorClientConfig.ServerName = ""
	aggregatorClientConfig.BearerToken = kubeClientConfig.BearerToken
	var aggregatorDiscoveryClient client.Interface
	err = wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		aggregatorClientConfig.Host = fmt.Sprintf("https://127.0.0.1:%d", atomic.LoadInt32(aggregatorPort))
		aggregatorDiscoveryClient, err = client.NewForConfig(aggregatorClientConfig)
		if err != nil {
			// this happens if we race the API server for writing the cert
			return false, nil
		}
		if _, err := aggregatorDiscoveryClient.Discovery().ServerVersion(); err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// now we're finally ready to test. These are what's run by defautl now
	testAPIGroupList(t, wardleClient.Discovery().RESTClient())
	testAPIGroup(t, wardleClient.Discovery().RESTClient())
	testAPIResourceList(t, wardleClient.Discovery().RESTClient())

	wardleCA, err := ioutil.ReadFile(wardleClientConfig.CAFile)
	if err != nil {
		t.Fatal(err)
	}
	aggregatorClient := aggregatorclient.NewForConfigOrDie(aggregatorClientConfig)
	_, err = aggregatorClient.ApiregistrationV1beta1().APIServices().Create(&apiregistrationv1beta1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.k8s.io"},
		Spec: apiregistrationv1beta1.APIServiceSpec{
			Service: &apiregistrationv1beta1.ServiceReference{
				Namespace: "kube-wardle",
				Name:      "api",
			},
			Group:                "wardle.k8s.io",
			Version:              "v1alpha1",
			CABundle:             wardleCA,
			GroupPriorityMinimum: 200,
			VersionPriority:      200,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// this is ugly, but sleep just a little bit so that the watch is probably observed.  Since nothing will actually be added to discovery
	// (the service is missing), we don't have an external signal.
	time.Sleep(100 * time.Millisecond)
	if _, err := aggregatorDiscoveryClient.Discovery().ServerResources(); err != nil {
		t.Fatal(err)
	}

	_, err = aggregatorClient.ApiregistrationV1beta1().APIServices().Create(&apiregistrationv1beta1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1."},
		Spec: apiregistrationv1beta1.APIServiceSpec{
			// register this as a loca service so it doesn't try to lookup the default kubernetes service
			// which will have an unroutable IP address since its fake.
			Group:                "",
			Version:              "v1",
			GroupPriorityMinimum: 100,
			VersionPriority:      100,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// this is ugly, but sleep just a little bit so that the watch is probably observed.  Since nothing will actually be added to discovery
	// (the service is missing), we don't have an external signal.
	time.Sleep(100 * time.Millisecond)
	_, err = aggregatorDiscoveryClient.Discovery().ServerResources()
	if err != nil {
		t.Fatal(err)
	}

	// TODO figure out how to turn on enough of services and dns to run more
}

func createKubeConfig(clientCfg *rest.Config) *clientcmdapi.Config {
	clusterNick := "cluster"
	userNick := "user"
	contextNick := "context"

	config := clientcmdapi.NewConfig()

	credentials := clientcmdapi.NewAuthInfo()
	credentials.Token = clientCfg.BearerToken
	credentials.ClientCertificate = clientCfg.TLSClientConfig.CertFile
	if len(credentials.ClientCertificate) == 0 {
		credentials.ClientCertificateData = clientCfg.TLSClientConfig.CertData
	}
	credentials.ClientKey = clientCfg.TLSClientConfig.KeyFile
	if len(credentials.ClientKey) == 0 {
		credentials.ClientKeyData = clientCfg.TLSClientConfig.KeyData
	}
	config.AuthInfos[userNick] = credentials

	cluster := clientcmdapi.NewCluster()
	cluster.Server = clientCfg.Host
	cluster.CertificateAuthority = clientCfg.CAFile
	if len(cluster.CertificateAuthority) == 0 {
		cluster.CertificateAuthorityData = clientCfg.CAData
	}
	cluster.InsecureSkipTLSVerify = clientCfg.Insecure
	config.Clusters[clusterNick] = cluster

	context := clientcmdapi.NewContext()
	context.Cluster = clusterNick
	context.AuthInfo = userNick
	config.Contexts[contextNick] = context
	config.CurrentContext = contextNick

	return config
}

func readResponse(client rest.Interface, location string) ([]byte, error) {
	return client.Get().AbsPath(location).DoRaw()
}

func testAPIGroupList(t *testing.T, client rest.Interface) {
	contents, err := readResponse(client, "/apis")
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiGroupList metav1.APIGroupList
	err = json.Unmarshal(contents, &apiGroupList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", "/apis", err)
	}
	assert.Equal(t, 1, len(apiGroupList.Groups))
	assert.Equal(t, groupVersion.Group, apiGroupList.Groups[0].Name)
	assert.Equal(t, 1, len(apiGroupList.Groups[0].Versions))
	assert.Equal(t, groupVersionForDiscovery, apiGroupList.Groups[0].Versions[0])
	assert.Equal(t, groupVersionForDiscovery, apiGroupList.Groups[0].PreferredVersion)
}

func testAPIGroup(t *testing.T, client rest.Interface) {
	contents, err := readResponse(client, "/apis/wardle.k8s.io")
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiGroup metav1.APIGroup
	err = json.Unmarshal(contents, &apiGroup)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", "/apis/wardle.k8s.io", err)
	}
	assert.Equal(t, groupVersion.Group, apiGroup.Name)
	assert.Equal(t, 1, len(apiGroup.Versions))
	assert.Equal(t, groupVersion.String(), apiGroup.Versions[0].GroupVersion)
	assert.Equal(t, groupVersion.Version, apiGroup.Versions[0].Version)
	assert.Equal(t, apiGroup.PreferredVersion, apiGroup.Versions[0])
}

func testAPIResourceList(t *testing.T, client rest.Interface) {
	contents, err := readResponse(client, "/apis/wardle.k8s.io/v1alpha1")
	if err != nil {
		t.Fatalf("%v", err)
	}
	t.Log(string(contents))
	var apiResourceList metav1.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", "/apis/wardle.k8s.io/v1alpha1", err)
	}
	assert.Equal(t, groupVersion.String(), apiResourceList.GroupVersion)
	assert.Equal(t, 1, len(apiResourceList.APIResources))
	assert.Equal(t, "flunders", apiResourceList.APIResources[0].Name)
	assert.True(t, apiResourceList.APIResources[0].Namespaced)
}

const (
	policyCachePollInterval = 100 * time.Millisecond
	policyCachePollTimeout  = 5 * time.Second
)
